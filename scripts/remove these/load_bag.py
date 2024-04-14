import rosbag
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
# Define the path to the ROS bag file
bag_file = '/home/bartonlab-user/workspace/src/graspmixer_demo/2024-02-18-13-27-00.bag'

# Define the output folders for the images
depth_folder = 'depth_raw_images'
rgb_folder = 'rgb_raw_images'

# Create output folders if they don't exist
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(rgb_folder, exist_ok=True)

# Initialize CvBridge
bridge = CvBridge()

# Open the bag file
with rosbag.Bag(bag_file, 'r') as bag:
    # Iterate through messages in the bag file
    for topic, msg, t in bag.read_messages():
        # Check if the topic is one of the desired topics
        if topic == '/depth/image_raw':
            # Convert ROS Image message to OpenCV 
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth_filename = os.path.join(depth_folder, f"{t.to_nsec()}.png")
            cv2.imwrite(depth_filename, cv_image)
            print(f"Saved depth image: {depth_filename}")
        
        elif topic =='/rgb_to_depth/image_raw':
            # Convert ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Save the RGB image
            rgb_filename = os.path.join(rgb_folder, f"{t.to_nsec()}.jpg")
            cv2.imwrite(rgb_filename, cv_image)
            print(f"Saved RGB image: {rgb_filename}")
