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

"""
Set intrinsic parameters for Microsoft Azure Depth camera
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

"""

def preprocess_point_cloud(pcd, voxel_size):
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

def draw_pointclouds(source, target, tf_source=None):
    if tf_source is None:
        tf_source = np.eye(4)
    source_temp = deepcopy(source).transform(tf_source)
    target_temp = deepcopy(target)
    o3d.visualization.draw_geometries([source_temp, target_temp, coordinate_frame])


#camera intrinsic parameters 
fx = 505.103
fy = 505.2
cx = 325.034
cy = 339.758
width = 640
height = 576
coordinate_frame = None

class ImageProcessor:
    """
        Initializes the ImageProcessor object. Sets up ROS subscribers for RGB and depth images,
        initializes variables for storing images and point clouds, and prints a message indicating
        that the topics have been subscribed to.
    """
    def __init__(self):
        self.bridge = CvBridge()
        # self.color_sub = rospy.Subscriber("/kmriiwa/rgb_to_depth/image_raw", Image, self.color_callback)
        # self.depth_sub = rospy.Subscriber("/kmriiwa/depth/image_raw", Image, self.depth_callback)
        self.color_sub = rospy.Subscriber("/rgb_to_depth/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/depth/image_raw", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None

    def color_callback(self, data):
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

def extract_rpy_xyz(transformation):
    R = transformation[:3,:3]
    T = transformation[:3,3]
    #embed()
    yaw = math.atan2(R[1,0],R[0,0])
    pitch = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = math.atan2(R[2,1], R[2,2])
    x,y,z = T
    return roll,pitch,yaw, x, y, z
        

def main():
    global coordinate_frame
    # Get color and depth image
    proc = ImageProcessor()
    while proc.color_image is None or proc.depth_image is None:
        rospy.sleep(0.1)
    color_image = proc.color_image
    depth_image = proc.depth_image

    ### Segmentation
    # Get mask from color image based on HSV
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([51,17,0])   #values to segment black color object (object 6)
    upper_bound = np.array([155,255,93]) 
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)
    x1,y1, x2, y2 =  180, 48, 546, 521  #can move the object, mask is for the whole 'workspace'
    mask_focused = np.zeros(contour_mask.shape[:2], dtype = np.uint8)
    mask_focused[y1:y2, x1:x2] = 255
    #mask_focused = contour_mask
    masked_image = cv2.bitwise_and(contour_mask, contour_mask, mask = mask_focused)

    # Apply mask to depth to get segmented depth
    segmented_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=masked_image)
    
    # Convert to open3d images
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(segmented_depth_image.astype(np.float32))

    # Convert to RGBD datastructure
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    camera_instrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    ### Create target and source point clouds
    # Convert observed RGBD to source point cloud
    source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_instrinsics)

    # Filter source point cloud
    filtered_source = source    # we were able to remove the filtering logic

    # Import target point cloud
    stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_06.stl'
    #stl_file = '/home/bartonlab-user/Desktop/output_models/large_obj_04.stl'
    mesh = o3d.io.read_triangle_mesh(stl_file)
    # mesh_scaled_centered = deepcopy(mesh).scale(0.001, center=mesh.get_center()) #because source is in mm, so scaled down to 0.001 from 1
    mesh_scaled = deepcopy(mesh).scale(0.001, center = 0*mesh.get_center()) 
    mesh = mesh_scaled
    # align it to the camera origin
    target = mesh.sample_points_uniformly(number_of_points=len(filtered_source.points))
    
    # Define coordinate frame for visualization purposes    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin = [0,0,0])

    # Visualize scene before transformations
    draw_pointclouds(source, target)

    ### Global registration
    # Define voxel size
    voxel_size = 0.001   # 1 mm

    # Preprocessing for GR
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Apply RANSAC
    ransac_dist_threshold = voxel_size * 10
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % ransac_dist_threshold)         

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
    draw_pointclouds(source, target, ransac_result.transformation)
    
    ### Local refinement
    # Guess RANSAC result initially
    initial_tf = deepcopy(ransac_result.transformation)
    embed()
    # ICP parameters
    print("=======Performing ICP===========")
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
        initial_tf[:3,-1] = np.zeros((3,))
        rotvec = np.random.uniform(-np.pi, np.pi, (3,))
        initial_tf[:3,:3] = Rot.from_rotvec(rotvec).as_matrix()

        icp_result = o3d.pipelines.registration.registration_icp(
                source, target, icp_dist_threshold, initial_tf,   
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000)
        )
        print(icp_result.fitness)
        if icp_result.fitness > 0.0:
            tform_.append(icp_result.transformation)
            rmse_.append(icp_result.inlier_rmse)
    
    rmse_ = np.array(rmse_)
    best_idx = np.argmin(rmse_)
    best_tform = tform_[best_idx]
    embed()
    # Visualize with ICP transformationf
    draw_pointclouds(source, target, best_tform)
    roll, pitch, yaw = Rot.from_matrix(deepcopy(best_tform)[:3,:3]).as_euler('zyx')
    roll, pitch, yaw, x, y, z = extract_rpy_xyz(np.linalg.inv(icp_result.transformation))
    print(x, y, z)
    embed()


if __name__ == "__main__":
    try:
        rospy.init_node('image_processor', anonymous=True)
        main()
    except Exception:
        print("Shutting down")





# import matplotlib.pyplot as plt
# plt.imshow(segmented_depth_image)
# plt.show()