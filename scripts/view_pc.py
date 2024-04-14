#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

rgb_image = None
depth_image = None

def rgb_callback(self,data):
    try:
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
        # Process the RGB image (e.g., display it)
        #cv2.imshow("RGB Image", cv_image)
       # cv2.waitKey(1)  # Adjust the delay as needed
        print("in rgb callback")
    except Exception as e:
        rospy.logerr(e)

def depth_callback(data):
    try:
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        # Process the depth image (e.g., display it)
        #cv2.imshow("Depth Image", cv_image)
        print("in depth callback")
#cv2.waitKey(1)  # Adjust the delay as needed
    except Exception as e:
        rospy.logerr(e)

def main():
    rospy.init_node('image_subscriber', anonymous=True)

    # Subscribe to the RGB topic
    rospy.Subscriber("/rgb/image_rect_color", Image, rgb_callback)

    # Subscribe to the depth topic
    rospy.Subscriber("/depth_to_rgb/hw_registered/image_rect", Image, depth_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()  
