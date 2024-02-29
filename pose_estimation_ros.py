import rospy
import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber("/color_topic", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/depth_topic", Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None
        self.source = None
        self.target = None


    def color_callback(self, data):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print(e)

    def process_images(self):
        if self.color_image is not None and self.depth_image is not None:
            
            segmented_mask = self.get_segmented_object(self.color_image)
            segmented_depth_image = cv2.bitwise_and(self.depth_image, self.depth_image, mask=segmented_mask)

            # Convert the segmented depth image to Open3D format
            color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(segmented_depth_image.astype(np.float32))


            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
                    )
            
            self.source = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, 
                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
            )
            
            self.filter_point_cloud()
            self.load_target_mesh()
            self.perform_icp_registration()
            

    def get_segmented_object(self, rgb_image):
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([92,76,35])   #values to segment object 6      
        upper_bound = np.array([111,255,255]) 
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        result = cv2.bitwise_and(rgb_image, rgb_image, mask = mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)
        
        return contour_mask

    def filter_point_cloud(self):
        min_distance = 0.01  
        max_distance = 50
        filtered_points = [point for point in np.asarray(self.source.points) if min_distance < np.linalg.norm(point) < max_distance]
        self.source = o3d.geometry.PointCloud()
        self.source.points = o3d.utility.Vector3dVector(filtered_points)

    def load_target_mesh(self):
        stl_file = '/home/bartonlab-user/workspace/src/graspmixer_demo/large_obj_06.stl'
        mesh = o3d.io.read_triangle_mesh(stl_file)
        mesh.scale(0.9, center=mesh.get_center())
        self.target = mesh.sample_points_uniformly(number_of_points=10000)

    def perform_icp_registration(self):
        threshold = 100
        initial_transformation = np.asarray([
            [0.862, 0.11, -0.507, 0.5], 
            [-0.139, 0.967, -0.215, 0.7],
            [0.487, 0.255, 0.835, -1.4],
            [0, 0, 0, 1]
        ])
        self.draw_registration_result(self.source, self.target, initial_transformation)

        print("Applying point to point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.source, self.target, threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
        )
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)

        print("Applying point to plane ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            self.source, self.target, threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
           o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
        )
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)




    def draw_registration_result(self,source, target, transformation):
        source_temp = copy.deepcopy(self.source)
        target_temp = copy.deepcopy(self.target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(self.transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom = 0.4459, 
                                        front = [0.9288, -0.2951, -0.2242],
                                        lookat = [1.6784, 2.0612, 1.44451],
                                        up =[-0.3402, -0.9189, -0.1996])

if __name__ == "__main__":
    rospy.init_node('image_processor', anonymous=True)
    processor = ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

