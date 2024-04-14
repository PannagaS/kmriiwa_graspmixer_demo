import rospy
import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from IPython import embed
from copy import deepcopy
import math
"""
Set intrinsic parameters for Microsoft Azure Depth camera
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

"""
#camera intrinsic parameters 
fx = 505.103
fy = 505.2
cx = 325.034
cy = 339.758
width = 640
height = 576

class ImageProcessor:
    """
        Initializes the ImageProcessor object. Sets up ROS subscribers for RGB and depth images,
        initializes variables for storing images and point clouds, and prints a message indicating
        that the topics have been subscribed to.
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/rgb_to_depth/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/depth/image_raw", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None
        self.source = None
        self.target = None
        self.best_target = None
        self.filtered_source = None
        self.timer = rospy.Timer(rospy.Duration(1/10), self.timer_callback)
 
    def timer_callback(self, event):
        if self.color_image is not None and self.depth_image is not None:
            self.process_images()


    def color_callback(self, data):
        # TODO: revisit these callbacks. 
        # In general they should only update class variables (self.color_image = ...)
        # and not do processing.
        """
        Callback function for the RGB image subscriber. Converts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imwrite("grey_box.png", self.color_image)
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        """
        Callback function for the depth image subscriber. Co0.25765707]nverts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            
        except CvBridgeError as e:
            print(e)

    def process_images(self):
        """
        Processes the RGB and depth images to perform object segmentation, point cloud creation, filtering,
        registration, and visualization.
        """
        if self.color_image is None or self.depth_image is None:
            return
        
        print("inside processing")
        
        
        segmented_mask = self.get_segmented_object(self.color_image)
        segmented_depth_image = cv2.bitwise_and(self.depth_image, self.depth_image, mask=segmented_mask)
        
        color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(segmented_depth_image.astype(np.float32))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
        camera_instrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        self.source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_instrinsics)

        self.filtered_source = self.filter_source_point_cloud(self.source)
        

        print("Number of points in filtered source : ", self.filtered_source) 
        
        self.target = self.load_target_mesh()

        #Compute and Align to scale
        #self.filtered_source, self.target = self.compute_and_align_to_scale(self.filtered_source, self.target) #rn the scale factor has been set to 1 as I'm downscaling in load_target_mesh
        

        #camera coordinates
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin = [0,0,0])
        #o3d.visualization.draw_geometries([self.filtered_source, self.target, self.coordinate_frame])

        #voxel_size = 0.015 #1.5cm
        voxel_size = 0.015 #1.5 cm
        self.filtered_source, self.target, self.filtered_source_down, self.target_down, self.filtered_source_fpfh, self.target_fpfh = self.prepare_dataset(voxel_size, self.filtered_source, self.target)

        #embed()
        result_ransac = self.execute_global_registration(self.filtered_source_down,  self.target_down, self.filtered_source_fpfh, self.target_fpfh, voxel_size)
        print(" Transformation obtained from Global Registration :\n ", result_ransac.transformation)
        #embed() #after GR

        #visualize GR with bounding box
        self.bbx_t = o3d.geometry.OrientedBoundingBox.get_axis_aligned_bounding_box(self.target_down)
        self.bbx_t.color = (0,0,0)
        #o3d.visualization.draw_geometries([self.filtered_source_down, self.target_down, self.bbx_t])
        #self.draw_registration_result(self.filtered_source_down, self.target_down, result_ransac.transformation) #visualize global registration result
        # self.draw_registration_result(self.filtered_source, self.target, result_ransac.transformation)
        # embed()
        num_initializations = 5
        result_icp, self.best_target = self.random_initializations(num_initializations, self.filtered_source, self.target, result_ransac.transformation, voxel_size)

        print("result icp transformation - ")
        print(result_icp.transformation)

        #result_icp = self.refine_registration(self.filtered_source, self.target, result_ransac.transformation)


        #self.target_centroid = result_icp.transformation @ self.target_centroid
        #self.target_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.target_centroid[:3])
        print("\n\n")
        print("*********************FINAL TRANSFORMATION***************************")
        print(result_icp) #prints metrics
        print(result_icp.transformation)
         
        roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(result_icp.transformation)
        print(f"Roll = {math.degrees(roll)}, Pitch = {math.degrees(pitch)}, Yaw = {math.degrees(yaw)}, x = {x}, y = {y}, z = {z}")

        print("Inverse of ICP result")
       
        self.T_inv = np.linalg.inv(result_icp.transformation)  # print(self.T_inv)

        roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(self.T_inv)
        print(f"Roll = {math.degrees(roll)}, Pitch = {math.degrees(pitch)}, Yaw = {math.degrees(yaw)}, x = {x}, y = {y}, z = {z}")
        #self.draw_registration_result(self.filtered_source, self.target, result_ransac.transformation)  #visualize global registration
        embed()
        self.draw_registration_result(self.filtered_source, self.target, result_icp.transformation)  #visualize icp result
        #self.draw_registration_result(self.filtered_source, self.target, self.T_inv ) #visualize inverted transformation

        #embed()
        print("==== Done ====")


    def random_initializations(self, num_initializations, filtered_source, target, transformation, voxel_size):
        rotated_pcds =[]
        bbx = []
        final_icp = None
        best_icp = None
        best_rmse = 9999
        best_fitness =0
        best_target = None
        #embed()
        # target = deepcopy(target).transform(transformation)
        # for i in range(num_initializations):
        #     angles = np.random.uniform(0,2*np.pi, 3)
        #     R_x = o3d.geometry.get_rotation_matrix_from_xyz((angles[0], 0,0))
        #     R_y = o3d.geometry.get_rotation_matrix_from_xyz((0,angles[1],0))
        #     R_z = o3d.geometry.get_rotation_matrix_from_xyz((0,0,angles[2]))
        #     rotation_matrix = R_z @ R_y @ R_x
        #     #####
        #     tmp = deepcopy(target)
        #     rotated_pcd = tmp.rotate(rotation_matrix, center = tmp.get_center())
        #     rotated_pcds.append(rotated_pcd)
        #     bbox = o3d.geometry.OrientedBoundingBox.get_oriented_bounding_box(rotated_pcd)
        #     bbox.color = (0,0,0)
        #     bbx.append(bbox)
        #     #embed()
             
        #     #####
        # target = deepcopy(target).transform(transformation)
        #     # extended_rotation_mat = np.eye(4)
        #     # extended_rotation_mat[:3, :3] = rotation_matrix
        #     # rotated_target = target.rotate(rotation_matrix)
        #     # combined_transformation = np.dot(transformation, extended_rotation_mat)
        #     icp = self.refine_registration(filtered_source, rotated_pcd, transformation, voxel_size)
        #     #embed()
        #     #make  4x4  [R;t] matrix
        #     # r_h = np.vstack((rotation_matrix, [[0, 0, 0]]))
        #     # t_h = np.array([ [transformation[0][3]], [transformation[1][3]],[ transformation[2][3]], [1]  ])
        #     # R = np.hstack((r_h, t_h))

        #     #icp = self.refine_registration(filtered_source, target, R, voxel_size)
        #     print(f"inliler_rmse in iteration {i} is {icp.inlier_rmse}")
        #     if icp.inlier_rmse < best_rmse:
        #         best_rmse = icp.fitness
        #         best_transformation = icp.transformation
        #         best_icp = icp
        #         best_target = rotated_pcd
        #         print("Rotated pcd = ", rotated_pcd)

        # final_list = []
        # final_list = rotated_pcds + [filtered_source] + bbx
        
       
        #o3d.visualization.draw_geometries(final_list)
        icp = self.refine_registration(filtered_source, target, transformation, voxel_size)
        #embed()
        #best_icp = icp
        # print("Inside random icp function ")
        # print(best_icp.transformation)
        best_icp = icp
        print("===========")
        return best_icp, best_target

    def rotate_point_cloud(self, pcd, rotation_matrix):
        centroid = pcd.get_center()
        pcd_temp = copy.deepcopy(pcd).translate(-centroid)
        pcd_temp.rotate(rotation_matrix, center = centroid)
        return pcd_temp.translate(centroid)

    def align_major_axis_to_z(self, pcd):
        points = np.asarray(pcd.points)
        mean = np.mean(points, axis=0)
        cov_matrix = np.cov(points - mean, rowvar=False)

        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        major_axis = eig_vec[:, np.argmax(eig_val)]

        z_axis = np.array([0,0,1])
        rotation_axis = np.cross(major_axis, z_axis)
        rotation_angle = np.arccos(np.dot(major_axis, z_axis))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis*rotation_angle)

        pcd.rotate(rotation_matrix, center=mean)
        return pcd


    def extract_rpy_xyz(self, transformation):
        R = transformation[:3,:3]
        T = transformation[:3,3]
        #embed()
        yaw = math.atan2(R[1,0],R[0,0])
        pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = math.atan2(R[2,1], R[2,2])
        x,y,z = T
        return roll,pitch,yaw, x, y, z
            

    def refine_registration(self, source, target, initial_transformation, voxel_size):
        """
        Refines the registration between the source and target point clouds using the Iterative Closest Point (ICP) algorithm.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        initial_transformation (np.ndarray): The initial transformation matrix.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
        """
        distance_threshold = voxel_size*0.005
        print("Applying ICP")
        print(" distance threshold = ", distance_threshold)

        # tform_ransac = initial_transformation
        # initial_transformation = np.eye(4)

        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #      source, target, 0.02, np.linalg.inv(initial_transformation),   #0.05
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(  max_iteration = 10000)
        # )
        reg_p2p = o3d.pipelines.registration.registration_icp(
             source, target, distance_threshold, initial_transformation,   #0.05
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria( max_iteration = 10000)
        )

     
        return reg_p2p

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        """
        Performs global registration between downsampled source and target point clouds using RANSAC and feature matching.

        Parameters:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the downsampled source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the downsampled target point cloud.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the global registration.
        """

        print("================INSIDE GLOBAL REGISTRATION========================")
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        
        # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #         source_down, target_down, source_fpfh, target_fpfh, True,
        #         distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        #       3, [
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
        #                 0.75),
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
        #                 distance_threshold)
        #         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.999))

        best_result = None
        best_fitness = 0
        num_initializations = 10
        centroid_distance_threshold = 0.01 #1cm
        centroid1 = np.mean(np.asarray(source_down.points), axis =0)
        centroid2 = np.mean(np.asarray(target_down.points), axis =0)
        #embed()
        #for i in range(num_initializations):
        centroid1 = np.mean(np.asarray(source_down.points), axis =0)
        initial_transformation = np.eye(4)
        initial_transformation[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, 3))
        initial_transformation[:3,-1] = centroid1 # initialize at centroid
         
        # target_down = target_down.transform(initial_transformation)

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
        return result
        
        # while(True):
        #     initial_transformation = np.eye(4)
        #     initial_transformation[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, 3))
        #     initial_transformation[:3,-1] = centroid1 # initialize at centroid
        #     target_down_gr = deepcopy(target_down)
        #     target_down_gr = target_down.transform(initial_transformation)
        #     o3d.visualization.draw_geometries([source_down, target_down_gr])
            
            
        #     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #         source_down, target_down_gr, source_fpfh, target_fpfh, True,
        #         distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        #         5, [
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
        #                 0.9),
        #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
        #                 distance_threshold)
        #         ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
            
            
            
        #     tform = result.transformation

        #     #target_down_gr.transform(tform)     # TODO: triple check that this transform is applied in the correct direction (not inverse)
        #     centroid2 = np.mean(np.asarray(target_down_gr.points), axis=0)
        #     d = np.linalg.norm(centroid1-centroid2) 
        #     print(d)


        #     # if result.fitness > best_fitness:
        #     #     best_fitness = result.fitness
        #     #     best_result = result

        #     if d <= 0.05:
        #         best_result = result
        #         break
        #     # embed()
        #     # best_result = result # TODO: uncomment the 5 cm check as before again
        #     break

        return best_result

    def prepare_dataset(self, voxel_size, source, target):
        """
        Prepares the dataset for registration by transforming the source point cloud, downsampling both point clouds,
        and computing FPFH features.

        Parameters:
        voxel_size (float): The voxel size used for downsampling.
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        centroid (np.ndarray): The centroid of the source point cloud.

        Returns:
        tuple: A tuple containing the source point cloud, target point cloud, downsampled source point cloud,
               downsampled target point cloud, FPFH feature of the downsampled source point cloud, and FPFH feature
               of the downsampled target point cloud.
        """

        print("=============INSIDE PREPARE DATASET===================")
        print(":: Load two point clouds and disturb initial pose.")
      
        #target.transform( np.identity(4))

        # centroid1 = np.mean(np.asarray(source.points), axis =0)
        # initial_transformation = np.eye(4)
        # initial_transformation[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, 3))
        # initial_transformation[:3,-1] = centroid1# initialize at centroid

        #have some z offset 
        #initial_transformation[3,3]+= 0.1 #move 4 cm above
        #target.transform(initial_transformation)
        #self.draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        Preprocesses a point cloud by downsampling, estimating normals, and computing FPFH features.

        Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to preprocess.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        tuple: A tuple containing the downsampled point cloud and its FPFH feature.
        """
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
       
    def compute_and_align_to_scale(self, filtered_source, target):
        """
        Computes the scale factor between the source and target point clouds and aligns them accordingly
        (currently unused as the scale factor is set to 1).

        Parameters:
        filtered_source (o3d.geometry.PointCloud): The filtered source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.

        Returns:
        tuple: A tuple containing the scaled source point cloud and the target point cloud.
        """
        aabb_source = filtered_source.get_axis_aligned_bounding_box()
        aabb_extent_source = aabb_source.get_extent()
        print("AABB source : ", aabb_extent_source)

        aabb_target = target.get_axis_aligned_bounding_box()
        aabb_extent_target = aabb_target.get_extent() 
        print("AABB target : ", aabb_extent_target)
        #print("source x = {}, y = {}, z = {}".format(aabb_extent_source[0], aabb_extent_source[1], aabb_extent_source[2]))
        #scale_factor = aabb_extent_target[0]/aabb_extent_source[0]  # target_  / source_
        scale_factor = 1

        scaled_source_pcd = filtered_source.scale(scale_factor, center = filtered_source.get_center())
        filtered_source = scaled_source_pcd
        aabb_after_scale = filtered_source.get_axis_aligned_bounding_box()
        aabb_after_scaling = aabb_after_scale.get_extent()
        print("Source after scaling  : ", aabb_after_scaling)
        return filtered_source, target

    def get_segmented_object(self, rgb_image):
        """
        Segments the object of interest from the RGB image using HSV color space and returns a binary mask.

        Parameters:
        rgb_image (np.ndarray): The RGB image to segment.

        Returns:
        np.ndarray: The binary mask of the segmented object.
        """
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([51,17,0])   #values to segment black color object (object 6)
        upper_bound = np.array([155,255,93]) 
        # lower_bound = np.array([92,76,35])   #values to segment blue color object (object 6)     
        # upper_bound = np.array([111,255,255]) 
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(rgb_image, rgb_image, mask = mask)

        

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)


        x1,y1, x2, y2 =  180, 48, 546, 521  #can move the object, mask is for the whole 'workspace'
        mask_focused = np.zeros(contour_mask.shape[:2], dtype = np.uint8)
        mask_focused[y1:y2, x1:x2] = 255
        
        masked_image = cv2.bitwise_and(contour_mask, contour_mask, mask = mask_focused)
        
        return masked_image

    def filter_source_point_cloud(self, source):
        """
        Filters the source point cloud by removing points that are too close or too far from the camera.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud to be filtered.

        Returns:
        o3d.geometry.PointCloud: The filtered source point cloud.
        """
         
        min_distance = 0.0 # Adjust this value as needed
        max_distance = 0.40  #0.35 for obj 6
        # Filter out points that are too close to the camera origin and beyond max_distance
        filtered_points = []
        for point in np.asarray(source.points):
            distance = np.linalg.norm(point)
            if min_distance < distance < max_distance:
                filtered_points.append(point)

        
        filtered_source = o3d.geometry.PointCloud()
        filtered_source.points = o3d.utility.Vector3dVector(filtered_points)
        
        # print("********************** Visualizing filtered pcd ********************")
        # o3d.visualization.draw_geometries([self.filtered_source])
        print("In filtering")
        print(filtered_source.points)
        return filtered_source

    def load_target_mesh(self):
        """
        Loads the target mesh from an STL file, scales it, samples points from it, and returns the resulting point cloud.

        Returns:
        o3d.geometry.PointCloud: The point cloud sampled from the target mesh.
        """
        stl_file = '/home/bartonlab-user/workspace/src/graspmixer_demo/large_obj_06.stl'
        #stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_04.stl'
        mesh = o3d.io.read_triangle_mesh(stl_file)
        # mesh_scaled_centered = deepcopy(mesh).scale(0.001, center=mesh.get_center()) #because source is in mm, so scaled down to 0.001 from 1
        mesh_scaled = deepcopy(mesh).scale(0.001, center = 0*mesh.get_center()) 
        mesh = mesh_scaled
        # align it to the camera origin

        self.target = mesh.sample_points_uniformly(number_of_points=len(self.filtered_source.points))
 
        return self.target

    def perform_icp_registration(self, filtered_source, target):
        threshold =10
        # initial_transformation = np.asarray([
        #     [0.862, 0.11, -0.507, 0.5], 
        #     [-0.139, 0.967, -0.215, 0.7],
        #     [0.487, 0.255, 0.835, -1.4],
        #     [0, 0, 0, 1]
        # ])
        initial_transformation = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]])
        #self.draw_registration_result(self.source, self.target, initial_transformation)
        print("************************************************")
        print("Applying point to point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            filtered_source, target,  threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria( relative_rmse = 1e-6, relative_fitness = 1e-6, max_iteration = 100)
        )
         

        print(reg_p2p)
        print("Point to Point transformation is:")
        
        print(reg_p2p.transformation)
        self.draw_registration_result(filtered_source, target, reg_p2p.transformation)
        o3d.visualization.draw_geometries([filtered_source, target, self.coordinate_frame])
        print("==========================================")


        # print("Applying point to plane ICP")
        # reg_p2l = o3d.pipelines.registration.registration_icp(
        #     filtered_source, target, threshold, initial_transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        # )
        # print(reg_p2l)
        # print("Point to Plane transformation is:")
        # print(reg_p2l.transformation)
        # self.draw_registration_result(filtered_source, target, reg_p2l.transformation)




    def draw_registration_result(self, filtered_source, target, transformation):
        """
        Visualizes the registration result by applying the transformation to the source or target point cloud
        and displaying them alongside a coordinate frame.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The transformation matrix to be applied.
        """
        source_temp = copy.deepcopy(filtered_source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

        # # Bring the depth towards the model 
        # source_temp_tf = deepcopy(source_temp).transform(transformation)
        # o3d.visualization.draw_geometries([source_temp_tf, target_temp, self.coordinate_frame])

        # Bring the model towards the depth
        tf_inv = np.linalg.inv(transformation)
        #target_temp_tf = target_temp.transform(tf_inv)

        #embed()

        target_tf = target_temp.transform(transformation)
        
        o3d.visualization.draw_geometries([source_temp, target_tf, self.coordinate_frame])


if __name__ == "__main__":
    rospy.init_node('image_processor', anonymous=True)
    processor = ImageProcessor()
    #processor.run()
    try:
        rospy.spin()
    except KeyboardInterrupter:
        print("Shutting down")