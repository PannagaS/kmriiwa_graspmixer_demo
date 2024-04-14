#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.colorImg_count =0
        self.depthImg_count = 0
        self.maxImg =10

        self.colorImg_sub = rospy.Subscriber("/rgb_to_depth/image_raw", Image, self.colorImg_callback)
        self.depthImg_sub = rospy.Subscriber("/depth/image_raw", Image, self.depthImg_callback)

    def colorImg_callback(self,data):
        if self.colorImg_count<self.maxImg:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data,'bgr8')
            except Exception as e:
                rospy.logerr(e)
            cv2.imwrite(f'/home/bartonlab-user/workspace/src/graspmixer_demo/color/rgb_frame{self.colorImg_count}.png',cv_image)
            self.colorImg_count +=1

    def depthImg_callback(self,data):
        if self.depthImg_count<self.maxImg:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data,'passthrough')
            except Exception as e:
                rospy.logerr(e)
            cv2.imwrite(f'/home/bartonlab-user/workspace/src/graspmixer_demo/depth/depth_frame{self.depthImg_count}.jpeg',cv_image)
            self.depthImg_count +=1

def main():
    rospy.init_node('image_saver', anonymous=True)
    image_saver = ImageSaver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
    