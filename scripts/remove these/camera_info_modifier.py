#! /usr/bin/env python
import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_callback(msg):
    modified_msg = msg
    modified_msg.D = modified_msg.D[:5]
    pub.publish(modified_msg)

rospy.init_node('camera_info_modifier')
rospy.Subscriber('/azure/rgb/camera_info', CameraInfo, camera_info_callback)
pub = rospy.Publisher('/modified_camera/info', CameraInfo, queue_size = 10)
rospy.spin()