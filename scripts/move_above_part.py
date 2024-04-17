#!/usr/bin/env python3

import rospy
from kmriiwa_commander import kmriiwa
from IPython import embed
from geometry_msgs.msg import Pose, Twist, PoseStamped
import tf_conversions as tfc
import numpy as np
from pose_estimation import PoseEstimation
import tf.transformations as tf_trans  
from tf import TransformListener
from tf import TransformListener, TransformBroadcaster
import tf2_ros
import tf2_geometry_msgs
import find_grasps_class
'''
Simple demo to move end-effector above perceived object.

29 March 2024
'''

#home position for pose_estimation
CAPTURE_CFG = [-2.0511133351352115, 0.30588943110544914, 0.20184071264986378, -1.632826974474426, -0.13857694820588123, 1.2409260422055313, -0.2703440402877017]
z_offset = 0.038 # see if you have to keep an offset after you do grasp planning

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

def get_joint_angles_from_pose(move_group, pose):
    move_group.set_pose_target(pose)
    plan = move_group.plan()
    
    if plan:
        return plan.joint_trajectory.points[-1].positions
    else:
        return None
     
def timer_callback(event):
    if pose_obj_wrt_depth is not None or  object_in_base_f is not None:
        publish_transform(pose_obj_wrt_depth, "depth_camera_link", "object_pose_wrt_depth")
    
        publish_transform(object_in_base_f, "kmriiwa_base_link", "object_in_base")

def main():
    
    rospy.init_node('move_above_part')
    listener = TransformListener()
    part_pose_wrtc = PoseEstimation()  
    robot = kmriiwa() 
    tf_listener = TransformListener()

    obj_path_for_grasp_planner = "/home/bartonlab-user/workspace/src/kmr-iiwa-gripkit-cr-plus-l/test/4096_large_obj_04"
    fg = find_grasps_class(obj_path_for_grasp_planner)
    table_height = 0.7
    iters = 1
    rospy.sleep(1) 
    waypoints = []

    robot.move_cfg(CAPTURE_CFG)     # move to object pose estimation config
    tf_obj_wrt_depth = part_pose_wrtc.get_pose(draw_for_debug=True) 
    pose_obj_wrt_depth = tfc.toMsg(tfc.fromMatrix(tf_obj_wrt_depth))
    tf_obj_wrt_base = convert_pose_to_base_frame(pose_obj_wrt_depth, "depth_camera_link", "kmriiwa_base_link") 
    #tf_obj_wrt_base.position.z += z_offset
    timer = rospy.Timer(rospy.Duration(0.001), timer_callback)

    #find waypoints to reach grasp_pose
    waypoints, tf_tcp_wrt_obj = fg.get_grasp(tf_obj_wrt_base, table_height, iters)
    
    #follow waypoints
    for waypoint in range(len(waypoints)):
        robot.move(waypoint)
        if waypoint == waypoints[-1]:
            #adjust for z offset if you have to

            #grasp the object (close gripper)
            robot.close_gripper()
            
            #move
            robot.move_cfg(CAPTURE_CFG) 

    #embed()
    


if __name__ == "__main__":
    main()



