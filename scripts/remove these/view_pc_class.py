#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d 
import copy
import sys
import os
import matplotlib.pyplot as plt
from IPython import embed

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_plotly([source_temp, target_temp],
                                      zoom = 0.4459, 
                                      front = [0.9288, -0.2951, -0.2242],
                                      lookat = [1.6784, 2.0612, 1.44451],
                                      up =[-0.3402, -0.9189, -0.1996])

def get_segmented_object(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([105,54,123])   #values to segment object 6      
    upper_bound = np.array([179,255,255]) 
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask to draw contours
    contour_mask = np.zeros_like(mask)
    
    # Draw contours on the blank mask
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return contour_mask




def processing(rgb_image, depth_image):

    #color_img = cv2.imread("/Users/user/Downloads/05/rgb/0000.png")
    #depth_img = cv2.imread('/Users/user/Downloads/0000_1.jpeg')
    color_img = rgb_image
    depth_img = depth_image
    cv2.imshow('fig', color_img)
    
    segmented_mask = get_segmented_object(color_img)
    segmented_depth_image = cv2.bitwise_and(depth_img, depth_img, mask=segmented_mask)
    cv2.imwrite('/home/bartonlab-user/Desktop/segmented_depth_image.jpeg', segmented_depth_image)
    obj6_path = '/home/bartonlab-user/Desktop/output_models/large_obj_06.stl'
    embed()
    ############ OPEN 3d stuff ############
    #path = '/Users/user/Downloads/02'
    rgb_image = o3d.io.read_image("/Users/user/Downloads/05/rgb/0000.png")
    depth_image = o3d.io.read_image('/Users/user/Downloads/segmented_mask.jpeg')
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,convert_rgb_to_intensity = False)
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
                                                        o3d.camera.PinholeCameraIntrinsic(
                                                            o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault
                                                        ))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,convert_rgb_to_intensity = False)
    plt.subplot(1,2,1)
    plt.title('Color')
    plt.imshow(rgbd_image.color)
    plt.subplot(1,2,2)
    plt.title('Depth')
    plt.imshow(rgbd_image.depth)
    plt.show()
    source_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    scale_factor = 220000
    source_pcd.scale(scale_factor, center=source_pcd.get_center())
    #visualize source pc
     
    o3d.visualization.draw_plotly([source_pcd])
    scale_factor = 1
     
    stl_file = '/Users/user/Desktop/Work/Term 2/ROB 590/output_models/large_obj_06.stl'

    mesh = o3d.io.read_triangle_mesh(stl_file)
    # Apply scaling transformation to the vertices
    mesh.scale(scale_factor, center=mesh.get_center())

    # Generate point cloud from the scaled mesh
    target_pcd = mesh.sample_points_uniformly(number_of_points=10000)

    # source = o3d.io.read_point_cloud(source_pcd)
    # target = o3d.io.read_point_cloud(target_pcd)
    threshold = 0.1

    trans_init = np.asarray([[0.862, 0.11, -0.507, 0.5], 
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4],
                            [0, 0, 0, 1]])
    
    source = source_pcd
    target = target_pcd 
    draw_registration_result(source, target, trans_init)

    print("initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    #applying point to point ICP
    print("Applying point to point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(source,target,
                                                        threshold,trans_init,
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    print(reg_p2p)
    print("Transformation is : ")
    print(reg_p2p.transformation)

    draw_registration_result(source, target, reg_p2p.transformation)

    print("==============================")
    #Point to Plane ICP
    print("Applying point to place ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(source,target, threshold, trans_init,
                                                        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    print(reg_p2l)
    print("Transformation is :")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)


    





class CaptureImg():

    def __init__(self):
        self.rgb_image = None
        self.depth_image = None
        self.bridge = CvBridge()



        # Subscribe to the RGB topic
        rospy.Subscriber("/rgb/image_rect_color", Image, self.rgb_callback)

        # Subscribe to the depth topic
        rospy.Subscriber("/depth_to_rgb/hw_registered/image_rect", Image, self.depth_callback)

        self.pub = rospy.Publisher("debugging", Image)


    def rgb_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
        
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            # Process the RGB image (e.g., display it)
            #cv2.imshow("RGB Image", cv_image)
        # cv2.waitKey(1)  # Adjust the delay as needed
            print("in rgb callback")
        except Exception as e:
            rospy.logerr(e)
        # cv2.imshow("RGB image", self.rgb_image)
        # cv2.waitKey(0)
            
        if self.rgb_image is not None and self.depth_image is not None:
            self.image_received = True
            self.process_images()
            

    def depth_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            self.depth_image = self.bridge.imgmsg_to_cv2(data,desired_encoding="passthrough")
            # Process the depth image (e.g., display it)
            #cv2.imshow("Depth Image", cv_image)
            print("in depth callback")
            #cv2.waitKey(1)  # Adjust the delay as needed
        except Exception as e:
            rospy.logerr(e)
        
        #cv2.imshow("fig", self.depth_image)
        #cv2.waitKey(3)
            
        if self.rgb_image is not None and self.depth_image is not None:
            self.image_received = True
            self.process_images()
    
    def show_frame(self):
        print(type(self.rgb_image))
        
        if self.rgb_image != None:
            print("rgb not none")
            #cv2.imshow('fig', self.rgb_image)

            self.pub.publish(self.bridge.cv2_to_imgmsg(self.rgb_image, "bgr8"))
    
    def process_images(self):
        if self.image_received:
            print("Processing images...")
            # Call the processing function with received images
            processing(self.rgb_image, self.depth_image)
            self.image_received = False  # Reset the flag after processing
   

if __name__ == '__main__':
    rospy.init_node('image_subscriber', anonymous=True)

    capture = CaptureImg()
    capture.show_frame()
    # Keep the program running until interrupted
    rospy.spin()
    cv2.destroyAllWindows()
