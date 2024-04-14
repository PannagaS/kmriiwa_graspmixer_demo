#!/usr/bin/env python3

import rospy
from kmriiwa_commander import kmriiwa
from IPython import embed
from geometry_msgs.msg import Pose, Twist, PoseStamped
import tf_conversions as tfc
import numpy as np
from pose_estimation_globalRegistration_basic_class import PoseEstimation
import tf.transformations as tf_trans  
from tf import TransformListener

from tf import TransformListener, TransformBroadcaster
import tf2_ros
import tf2_geometry_msgs
'''
Simple demo to move end-effector above perceived object.

29 March 2024
'''

CAPTURE_CFG = [-2.0511133351352115, 0.30588943110544914, 0.20184071264986378, -1.632826974474426, -0.13857694820588123, 1.2409260422055313, -0.2703440402877017]
z_offset = 0.038 #4.8 cm

def publish_transform(pose, parent_frame, child_frame):
    broadcaster = TransformBroadcaster()
    translation = (pose.position.x, pose.position.y, pose.position.z)
    rotation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    broadcaster.sendTransform(translation, rotation, rospy.Time.now(), child_frame, parent_frame)


def convert_pose_to_base_frame(pose, from_frame, to_frame):
    """
    Convert a pose from one frame to another frame.

    Args:
        pose: The pose to be converted (geometry_msgs/Pose).
        from_frame: The source frame of the pose.
        to_frame: The target frame of the pose.

    Returns:
        The pose converted to the target frame.
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = from_frame
    pose_stamped.pose = pose

    try:
        # Wait for the transform to be available
        tf_listener.waitForTransform(to_frame, from_frame, rospy.Time(0), rospy.Duration(3.0))
        # Transform the pose
        transformed_pose = tf_listener.transformPose(to_frame, pose_stamped)
        return transformed_pose.pose
    except Exception as e:
        rospy.logerr(f"Failed to transform pose from {from_frame} to {to_frame}: {e}")
        return None

# def convert_pose_to_base_frame(input_pose, from_frame, to_frame):
#     tf_buffer = tf2_ros.Buffer()
#     listener = tf2_ros.TransformListener(tf_buffer)
    
#     pose_stamped = tf2_geometry_msgs.PoseStamped()
#     pose_stamped.pose = input_pose
#     pose_stamped.header.frame_id = from_frame
#     pose_stamped.header.stamp = rospy.Time.now()
#     try:
#         output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(0.1))
#         return output_pose_stamped.pose
#     except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
#         raise




def get_joint_angles_from_pose(move_group, pose):
    move_group.set_pose_target(pose)
    plan = move_group.plan()
    
    if plan:
        return plan.joint_trajectory.points[-1].positions
    else:
        return None
     

def timer_callback( event):
    if pose_obj_wrt_depth is not None or  object_in_base_f is not None:
        publish_transform(pose_obj_wrt_depth, "depth_camera_link", "object_pose_wrt_depth")
    
        publish_transform(object_in_base_f, "kmriiwa_base_link", "object_in_base")



rospy.init_node('move_above_part')

listener = TransformListener()


part_pose_wrtc = PoseEstimation()  
robot = kmriiwa() 
tf_listener = TransformListener()


rospy.sleep(1) 

robot.move_cfg(CAPTURE_CFG)     # move to object pose estimation config

# listener.waitForTransform('chassis_8', 'kmriiwa_base_link', rospy.Time(0), rospy.Duration(5))
# pq_obj_wrt_base = listener.lookupTransform('kmriiwa_base_link', 'chassis_8', rospy.Time(0))
# pose_obj_wrt_base = tfc.toMsg(tfc.fromTf(pq_obj_wrt_base))

# Get object pose
# tf_obj_wrt_cam = part_pose_wrtc.get_pose(draw_for_debug=True)  # exactly which frame in the tf tree? (rgb_camera_link)
# embed()
# dummy_rot = np.eye(3)
# dummy_rot[1,1] = -1
# dummy_rot[2,2] = -1
# tf_object_wrt_rgb = np.copy(tf_obj_wrt_cam)
# tf_object_wrt_rgb[:3, :3] = dummy_rot
# embed()
# pose_obj_wrt_cam = tfc.toMsg(tfc.fromMatrix(tf_obj_wrt_cam))
# object_in_base_f = convert_pose_to_base_frame(pose_obj_wrt_cam, "usb_cam_kmriiwa_wrist", "kmriiwa_base_link") 
# object_in_base_f.position.z += z_offset
tf_obj_wrt_depth = part_pose_wrtc.get_pose(draw_for_debug=True)  # exactly which frame in the tf tree? (rgb_camera_link)

pose_obj_wrt_depth = tfc.toMsg(tfc.fromMatrix(tf_obj_wrt_depth))
object_in_base_f = convert_pose_to_base_frame(pose_obj_wrt_depth, "depth_camera_link", "kmriiwa_base_link") 
object_in_base_f.position.z += z_offset
timer = rospy.Timer(rospy.Duration(0.001), timer_callback)
embed()
robot.move(object_in_base_f)
# pose_obj_wrt_base.position.z += 0.05
# robot.move(pose_obj_wrt_base)
embed()

robot.close_gripper()

robot.move_cfg(CAPTURE_CFG) 






