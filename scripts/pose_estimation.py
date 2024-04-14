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
from scipy.spatial.transform import Rotation as Rot

import tf_conversions as tfc
import tf.transformations as tf_trans  
from tf import TransformListener, TransformBroadcaster
import threading
class PoseEstimation:
    
    def __init__(self):
        """
        Initializes the ImageProcessor object. Sets up ROS subscribers for RGB and depth images,
        initializes variables for storing images and point clouds, and prints a message indicating
        that the topics have been subscribed to.
        """
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/kmriiwa/azure/rgb_to_depth/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/kmriiwa/azure/depth/image_raw", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None
        self.coordinate_frame = None

        #define intrinsics 
        self.fx = 505.103
        self.fy = 505.2
        self.cx = 325.034
        self.cy = 339.758
        self.width = 640
        self.height = 576
        

        #define stl path
        self.stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_06.stl'

        # self.publishing_thread = threading.Thread(target=self.start_publishing_transform)
        # self.publishing_thread.start()
        self.timer = rospy.Timer(rospy.Duration(0.001), self.timer_callback)

    def color_callback(self, data):
        """
        Callback function for the RGB image subscriber. Converts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #cv2.imwrite("grey_box.png", self.color_image)
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        """
        Callback function for the depth image subscriber. Converts the ROS image message to an OpenCV image,
        saves it, and calls process_images() to process the images if both RGB and depth images are available.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

    def extract_rpy_xyz(self, transformation):
        """
        Extract roll, pitch, yaw, and x, y, z from the 4x4 transformation matrix

        Parameters:
        transformation (np.float 2d array)
 
        Returns:
        roll, pitch, yaw, x, y, z
        """
        R = transformation[:3,:3]
        T = transformation[:3,3]

        yaw = math.atan2(R[1,0],R[0,0])
        pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = math.atan2(R[2,1], R[2,2])
        x,y,z = T
        return roll,pitch,yaw, x, y, z

    def preprocess_point_cloud(self, pcd, voxel_size):
        """
        Preprocesses a point cloud by downsampling, estimating normals, and computing FPFH features.

        Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud to preprocess.
        voxel_size (float): The voxel size used for downsampling.

        Returns:
        tuple: A tuple containing the downsampled point cloud and its FPFH feature.
        """
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def draw_pointclouds(self, source, target, tf_source=None):
        if tf_source is None:
            tf_source = np.eye(4)
        source_temp = deepcopy(source).transform(tf_source)
        target_temp = deepcopy(target)
        o3d.visualization.draw_geometries([source_temp, target_temp, self.coordinate_frame])
    
    def publish_transform(self, tform):
        pq_obj_wrt_cam = tfc.toTf(tfc.fromMatrix(tform))
        broadcaster = TransformBroadcaster()
        broadcaster.sendTransform(*pq_obj_wrt_cam, rospy.Time.now(), 'object_frame', 'rgb_camera_link')

    
   
    def timer_callback(self, event):
        if self.best_tform is not None:
            self.publish_transform(self.best_tform)

    def get_pose(self, draw_for_debug=False):
        # Wait for images
        """
        Get pose duh 
         
        """
        while self.color_image is None or self.depth_image is None:
            rospy.sleep(0.1)
        color_image = self.color_image
        depth_image = self.depth_image
        
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([51,17,0])   #values to segment black color object (object 6/ gray color)
        upper_bound = np.array([155,255,93]) 
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)
        #x1,y1, x2, y2 =  180, 48, 546, 521  #can move the object, mask is for the whole 'workspace'
        #mask_focused = np.zeros(contour_mask.shape[:2], dtype = np.uint8)
        #mask_focused[y1:y2, x1:x2] = 255
        #mask_focused = contour_mask #check if you want to modify this in the fu
        #masked_image = cv2.bitwise_and(contour_mask, contour_mask, mask = mask_focused)

        # Apply mask to depth to get segmented depth
        segmented_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=contour_mask)
        #embed()
        
        # Convert to open3d images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(segmented_depth_image.astype(np.float32))

        # Convert to RGBD datastructure
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
        camera_instrinsics = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

        """
        Create target and source point clouds
        """
        # Convert observed RGBD to source point cloud
        source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_instrinsics)

        # Filter source point cloud
        filtered_source = source    # we were able to remove the filtering logic

        embed()
        mesh = o3d.io.read_triangle_mesh(self.stl_file) #path to stl file is set in initialization
        mesh_scaled = deepcopy(mesh).scale(0.001, center = 0*mesh.get_center()) 
        mesh = mesh_scaled

        # align it to the camera origin
        target = mesh.sample_points_uniformly(number_of_points=len(filtered_source.points))
        
        # Define coordinate frame for visualization purposes    
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin = [0,0,0])

        # Visualize scene before transformations
        if draw_for_debug:
            self.draw_pointclouds(source, target)

        ### Global registration
        # Define voxel size
        voxel_size = 0.001   # 1 mm
        #flipping source and target
        #source, target = target, source


        # Preprocessing for GR
        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)

        # Apply RANSAC
        ransac_dist_threshold = voxel_size * 10
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % ransac_dist_threshold)         

        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                ransac_dist_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        ransac_dist_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))

        # Visualize with RANSAC transformation

        
        if draw_for_debug:
            self.draw_pointclouds(source, target, ransac_result.transformation)
        embed()
        ### Local refinement

        ###################
        
 



        ##################
        # Guess RANSAC result initially
        initial_tf = deepcopy(ransac_result.transformation)

        # ICP parameters
        # print("=======Performing ICP===========")
        icp_dist_threshold = voxel_size * 1000
        # Apply ICP
        # rotm_flip_x = Rot.from_rotvec([np.pi, 0, 0]).as_matrix()
        # rotm_init = initial_tf[:3,:3]
        # rotm_flip = rotm_flip_x @ rotm_init
        num_iters = 10
        tform_ = []
        rmse_ = []
        
        

        for i in range(num_iters):
            inital_tf = np.eye(4)
            # initial_tf[:3,-1] = np.zeros((3,)) #spawn at 0,0,0
            initial_tf[:3, -1] = source.get_center()
            rotvec = np.random.uniform(-np.pi, np.pi, (3,))
            initial_tf[:3,:3] = Rot.from_rotvec(rotvec).as_matrix()

            icp_result = o3d.pipelines.registration.registration_icp(
                    source, target, icp_dist_threshold, initial_tf,   
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000)
            )
            # print(icp_result.fitness)
            if icp_result.fitness > 0.0:
                tform_.append(icp_result.transformation)
                rmse_.append(icp_result.inlier_rmse)
        
        rmse_ = np.array(rmse_)
        best_idx = np.argmin(rmse_)
        best_tform = tform_[best_idx]
        #best_tform = icp_result.transformation
        # Visualize with ICP transformation
        if draw_for_debug:
            self.draw_pointclouds(source, target, best_tform)

        # roll, pitch, yaw = Rot.from_matrix(deepcopy(best_tform)[:3,:3]).as_euler('zyx')
        # roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(np.linalg.inv(icp_result.transformation))
        radius = 0.015
        x,y,z  = np.linalg.inv(best_tform)[:3, 3]
        highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius = radius) 
        highlight_sphere.translate([x,y,z])  
        highlight_sphere.paint_uniform_color([0,1,0]) 
        source_temp = deepcopy(source).transform(best_tform) 
        target_temp = deepcopy(target) 

        rotation_matrix = np.linalg.inv(best_tform)[:3,:3]
        rotation = Rot.from_matrix(rotation_matrix)
        translation = np.linalg.inv(best_tform)[:3, 3]
        rpy_angles = rotation.as_euler('xyz', degrees = False)
        body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
        body_frame.rotate(rotation_matrix, center=(0,0,0))
        body_frame.translate(translation) 

        o3d.visualization.draw_geometries([source, target, highlight_sphere, self.coordinate_frame, body_frame]) 

        #debugging
        #without inverse
        # rotation_matrix = np.copy(best_tform[:3,:3])
        # rotation = Rot.from_matrix(rotation_matrix)
        # translation = np.copy(best_tform[:3, 3])
        # rpy_angles = rotation.as_euler('xyz', degrees = False)
        # body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
        # body_frame.rotate(rotation_matrix, center=target.get_center())
        # body_frame.translate(translation) 


        # o3d.visualization.draw_geometries([source, target, highlight_sphere, self.coordinate_frame, body_frame]) 


        
        
        # dummy_rot = np.eye(3)
        # dummy_rot[1,1] = -1
        # dummy_rot[2,2] = -1
        # best_tform_hc = np.copy(best_tform)
        # translation = np.copy(best_tform[:3, 3])

        # best_tform_hc[:3, :3] = dummy_rot
        # body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
        # body_frame.rotate(best_tform_hc[:3, :3] , center=target.get_center())
        # body_frame.translate(translation) 

        
        # o3d.visualization.draw_geometries([source, target, highlight_sphere, self.coordinate_frame, body_frame])
        # ######debuggig end
        embed()
        self.best_tform = np.linalg.inv(best_tform)
        self.publish_transform(self.best_tform)
        return self.best_tform   # tform of obj wrt camera frame
        # return x, y, z

if __name__ == '__main__':
    rospy.init_node('pose_estimation_node')
    pose_estimator = PoseEstimation()
    rospy.spin()