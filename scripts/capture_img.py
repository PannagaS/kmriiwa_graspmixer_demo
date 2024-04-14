#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        # Save the image as PNG
        filename = rospy.Time.now().to_sec() + ".png"
        cv2.imwrite(filename, cv_image)
        rospy.loginfo("Saved image as %s" % filename)

if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
