#!/usr/bin/env python3

import rospy
from kmriiwa_commander import kmriiwa
from IPython import embed
from geometry_msgs.msg import Pose, Twist, PoseStamped
import tf_conversions as tfc
import numpy as np
import tf.transformations as tf_trans  
from tf import TransformListener, TransformBroadcaster


def main():
    rospy.init_node('manual_calibration_pub')

    listener = TransformListener()
    broadcaster = TransformBroadcaster()

    tcp_link = 'kmriiwa_tcp'
    usb_cam_link = 'usb_cam_kmriiwa_wrist'
    cam_base_link = 'camera_base'
    rgb_cam_link = 'rgb_camera_link'
    
    # Look up tf between TCP (e) and usb cam frame (u) (same as RGB link?) (should be static)
    listener.waitForTransform(tcp_link, usb_cam_link, rospy.Time(0), rospy.Duration(10))
    pq_r_wrt_e = listener.lookupTransform(tcp_link, usb_cam_link, rospy.Time(0))
    T_r_wrt_e = tfc.toMatrix(tfc.fromTf(pq_r_wrt_e))

    # Look up tf between camera body (c) and RGB link (r) (should be static)
    listener.waitForTransform(rgb_cam_link, cam_base_link, rospy.Time(0), rospy.Duration(10))
    pq_c_wrt_r = listener.lookupTransform(rgb_cam_link, cam_base_link, rospy.Time(0))
    T_c_wrt_r = tfc.toMatrix(tfc.fromTf(pq_c_wrt_r))

    # Compute tf between camera body (c) and TCP (e)
    T_c_wrt_e = T_r_wrt_e @ T_c_wrt_r
    pq_c_wrt_e = tfc.toTf(tfc.fromMatrix(T_c_wrt_e))

    ## Publish tf between camera body (c) and TCP (e)
    # Publish once and check error
    broadcaster.sendTransform(*pq_c_wrt_e, rospy.Time.now(), cam_base_link, tcp_link)

    # Check tf error between rgb link (r) and usb cam (u) (should be zero)
    listener.waitForTransform(rgb_cam_link, usb_cam_link, rospy.Time(0), rospy.Duration(10))
    pq_u_wrt_r = listener.lookupTransform(rgb_cam_link, usb_cam_link, rospy.Time(0))
    pos_err = np.linalg.norm(pq_u_wrt_r[0])
    quat_err1 = np.linalg.norm(np.array(pq_u_wrt_r[1]) - np.array([0,0,0,1]))
    quat_err2 = np.linalg.norm(np.array(pq_u_wrt_r[1]) - np.array([0,0,0,-1]))

    if pos_err > 0.001 or np.min([quat_err1, quat_err2]) > 0.001:
        raise Exception('Calibration failed! Check code in manual_calibration_pub.py')

    # Publish verified transform continuously
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        broadcaster.sendTransform(*pq_c_wrt_e, rospy.Time.now(), cam_base_link, tcp_link)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except:
        pass