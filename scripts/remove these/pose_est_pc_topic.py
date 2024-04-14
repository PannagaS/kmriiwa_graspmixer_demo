import rospy
import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from IPython import embed
from copy import deepcopy
import math
 

class ImageProcessor:
    """
        Initializes the ImageProcessor object. Sets up ROS subscribers for RGB and depth images,
        initializes variables for storing images and point clouds, and prints a message indicating
        that the topics have been subscribed to.
    """
    def __init__(self):
        
         

        self.pc_sub= rospy.Subscriber('/depth_to_rgb/points', PointCloud2 , self.pc_callback)
        
        self.source = None
        self.target = None
        self.best_target = None
        self.filtered_source = None
        self.timer = rospy.Timer(rospy.Duration(1/10), self.timer_callback)

    def pc_callback(self, data):
         
        try:
            #print('inside cb')
            points_list = list(pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")))
            points = np.array(points_list, dtype=np.float64)
            self.source = o3d.geometry.PointCloud()
            self.source.points = o3d.utility.Vector3dVector(points)
             
        except CvBridgeError as e:
            print(e)
 
    def timer_callback(self, event):
        if self.source is not None:
            self.process_point_cloud()
             

    

    def process_point_cloud(self):
        if self.source is not None:

            print("processing pc")
            
            #source point cloud is in self.source
            #load target pcd
            #self.filtered_source = self.filter_source_point_cloud(self.source) #no filtering done fn
            self.filtered_source = self.source
            
            self.target = self.load_target_mesh()

            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin = [0,0,0])
            embed()
            voxel_size = 0.05 #1.5 cm
            self.filtered_source, self.target, self.filtered_source_down, self.target_down, self.filtered_source_fpfh, self.target_fpfh = self.prepare_dataset(voxel_size, self.filtered_source, self.target)

            result_ransac = self.execute_global_registration(self.filtered_source_down,  self.target_down, self.filtered_source_fpfh, self.target_fpfh, voxel_size)
            
            # Visualize RANSAC
            source_tf = deepcopy(self.filtered_source).transform(result_ransac.transformation)
            o3d.visualization.draw_geometries([source_tf, self.target])             

            embed()
            result_icp = self.refine_registration(self.filtered_source, self.target, result_ransac.transformation, voxel_size)

            # Visualize ICP
            source_tf = deepcopy(self.filtered_source).transform(result_icp.transformation)
            o3d.visualization.draw_geometries([source_tf, self.target])

            print("\n\n")
            print("*********************FINAL TRANSFORMATION***************************")
            print(result_icp) #prints metrics
            print(result_icp.transformation)
            
            roll, pitch, yaw, x, y, z = self.extract_rpy_xyz(result_icp.transformation)
            print(f"Roll = {math.degrees(roll)}, Pitch = {math.degrees(pitch)}, Yaw = {math.degrees(yaw)}, x = {x}, y = {y}, z = {z}")
            self.draw_registration_result(self.filtered_source, self.target, result_icp.transformation)
            embed()
            print("========done=========")
           
    def load_target_mesh(self):
        """
        Loads the target mesh from an STL file, scales it, samples points from it, and returns the resulting point cloud.

        Returns:
        o3d.geometry.PointCloud: The point cloud sampled from the target mesh.
        """
        #stl_file = '/home/bartonlab-user/workspace/src/graspmixer_demo/obj_11.ply'
        stl_file = "/home/bartonlab-user/Desktop/output_models/large_obj_06.stl"

        #stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_04.stl'
        mesh = o3d.io.read_triangle_mesh(stl_file)
        # mesh_scaled_centered = deepcopy(mesh).scale(0.001, center=mesh.get_center()) #because source is in mm, so scaled down to 0.001 from 1
        mesh_scaled = deepcopy(mesh).scale(0.001, center = 0*mesh.get_center()) 
        mesh = mesh_scaled
        # align it to the camera origin

        #self.target = mesh.sample_points_uniformly(number_of_points=len(self.filtered_source.points))
        self.target = mesh.sample_points_uniformly(number_of_points=2048)
 
        return self.target


    def extract_rpy_xyz(self, transformation):
        R = transformation[:3,:3]
        T = transformation[:3,3]
        #embed()
        yaw = math.atan2(R[1,0],R[0,0])
        pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
        roll = math.atan2(R[2,1], R[2,2])
        x,y,z = T
        return roll,pitch,yaw, x, y, z

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

        centroid1 = np.mean(np.asarray(source.points), axis =0)
        initial_transformation = np.eye(4)

        # source.transform(initial_transformation)  # TODO: Tyler did this; undo

        # initial_transformation[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(-np.pi, np.pi, 3))
        # initial_transformation[:3,-1] = centroid1# initialize at centroid

        # #have some z offset 
        # initial_transformation[3,3]+= 0.1 
        # #initial_transformation = np.eye(4)
        # target = target.transform(initial_transformation)
        # embed()

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
        

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999))
        return result
        
       
    def filter_source_point_cloud(self, source):
        """
        Filters the source point cloud by removing points that are too close or too far from the camera.

        Parameters:
        source (o3d.geometry.PointCloud): The source point cloud to be filtered.

        Returns:
        o3d.geometry.PointCloud: The filtered source point cloud.
        """
         
        min_distance = 0.3 # Adjust this value as needed
        max_distance = 0.4#0.35 for obj 6
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
        distance_threshold = voxel_size * 0.2
        loss = o3d.pipelines.registration.TukeyLoss(k = 0.1)

        #distance_threshold = 0.01
        print("Applying ICP")
        print(" distance threshold = ", distance_threshold)

        # tform_ransac = initial_transformation
        # initial_transformation = np.eye(4)

        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #      source, target, 0.02, np.linalg.inv(initial_transformation),   #0.05
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(  max_iteration = 10000)
        # )
        print('::: INSIDE ICP')
        embed()
        reg_p2p = o3d.pipelines.registration.registration_icp(
             source, target, distance_threshold, initial_transformation,   #0.05
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        # Visualize ICP
        source_tf = deepcopy(source).transform(reg_p2p.transformation)
        o3d.visualization.draw_geometries([source_tf, target])

     
        return reg_p2p


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
  
    try:
        rospy.spin()
    except KeyboardInterrupter:
        print("Shutting down")