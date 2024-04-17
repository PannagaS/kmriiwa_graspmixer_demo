import os
import time
import rospy
import random
import argparse
import numpy as np
from tqdm import tqdm
import copy
from utils import Conversion
from env_manager import EnvManager
from robot_manager import Move_Robot
from IPython import embed
import tf.transformations as tf
from geometry_msgs.msg import Pose
from utils import Conversion, noisy_pose
from scipy.spatial.transform import Rotation as R



def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data




def grasp_gen(path: str = None, is_6dof: bool = False, marker: bool = False, pose: Pose=None) -> np.ndarray:
        """
        Generate grasp dictionary based on the current object pose and 
        contact point pairs (cpp).

        Parameters
        ----------
        path : string
            path to a .txt file of cpps

        Returns
        -------
        centers : 3xN : obj : `np.ndarray`
            array of potential gripper centers w.r.t the world frame
        directions : 3xN : obj : `np.ndarray`
            array of potential gripper directions w.r.t the world frame
        """
        with open(f"{path}/4096_large_obj_04_cpps.txt") as f:
            cpps = eval(f.read())
        cpps = np.asarray(cpps).T
        
        base2obj = Conversion().pose2T(pose)
        if is_6dof is True:
            with open(f"{path}/4096_large_obj_04_aprvs.txt") as f:
                aprvs = eval(f.read())
            for i in range(len(aprvs)):
                for j in range(len(aprvs[i])):
                    aprvs[i][j] = base2obj[:3,:3] @ aprvs[i][j]
        else:
            aprvs = None
        
        # array of potential gripper configurations w.r.t the object frame
        # 0.005 = 0.001(mm to m) * 0.5
        add_row = np.ones(cpps.shape[1])
        centers = 0.0005 * (cpps[:3,:] + cpps[3:,:])
        directions = cpps[:3,:] - cpps[3:,:]

        # array of potential gripper configurations w.r.t the world frame
        temp = base2obj @ np.vstack((centers, add_row))
        centers = temp[:3,:]
        directions = base2obj[:3,:3] @ directions
        if marker is True:
            with open(f"/home/bartonlab-user/workspace/src/kmr-iiwa-gripkit-cr-plus-l/test/4096_large_obj_04/4096_large_obj_04_scores.txt") as f:
                probs = eval(f.read())
            idx = [ind for ind, ele in enumerate(probs) if ele > 0.5]
            aprvs = [aprvs[i] for i in idx]
            probs = [probs[i] for i in idx]
            centers, directions = centers[:,idx], directions[:,idx]
        else:
            probs = None

        return centers, directions, aprvs, probs
 

def avoid_collision(table_height: float, centers: np.ndarray, directions: np.ndarray, aprvs: list = None) -> np.ndarray:
        """
        Remove grasp configurations that potentially causes collision with the table

        Parameters
        ----------
        directions : 3xN : obj : `np.ndarray`
            array of potential gripper directions w.r.t the world frame

        Returns
        -------
        total_list : 1xN : obj : `list`
            list indicating whether gripper centers and directions are selected
        num_selected : int
            total number of selected configurations
        """
        num_selected = 0
        refined_list = []
        for i in range(directions.shape[1]):
            # Check whether side of each gripper finger is potentially bellow the table surface (collision)
            ftip_loc1 = centers[:,i] + 0.007 * directions[:,i] / np.linalg.norm(directions[:,i])
            ftip_loc2 = centers[:,i] - 0.007 * directions[:,i] / np.linalg.norm(directions[:,i])

            if ftip_loc1[2] > table_height and ftip_loc2[2] > table_height:
                refined_list.append(i)
                num_selected = num_selected + 1
            elif aprvs is not None:
                refined_list.append(False)
            else:
                continue

        return refined_list, num_selected


def execute(pose, center: np.ndarray, direction: np.ndarray,  sub_aprvs: list = None,
                prob: list = None, repeat: int = None) -> float:
        """
        Execute a given grasp configuration # times to obtain its probability of success

        Parameters
        ----------
        center : 3xN : obj : `np.ndarray`
            array of potential gripper centers w.r.t the world frame
        direction : 3xN : obj : `np.ndarray`
            array of potential gripper directions w.r.t the world frame
        sdf_name : string
            sdf file of objects we spawn in the gazebo world
        repreat : int
            number of iteration to execute the grasp configuration

        Returns
        -------
        float : probability of successing the grasp
        """
        #mr = Move_Robot()
        if prob is not None:
            idx = prob.index(max(prob))
            center, direction, sub_aprvs = center[:,idx], direction[:,idx], sub_aprvs[idx]
        elif sub_aprvs is not None:
            random.seed()
            idx = random.randint(0, len(sub_aprvs) - 1)
            center, direction, sub_aprvs = center[:,idx], direction[:,idx], sub_aprvs[idx]

        attempt, total = 0, repeat
        
        for _ in range(repeat):
            time.sleep(1)
            grasp_pose = cartesian_space(waypoints=pose, tp_heights=[0.45, 0.3], sub_aprvs=sub_aprvs, center=center, direction=direction, top_down_flag=True)
            time.sleep(1)
             

            return grasp_pose
            


def cartesian_space( waypoints, tp_heights: list = None, 
                    center: np.ndarray = None, direction: np.ndarray = None, 
                    sub_aprvs: list = None, top_down_flag: bool = True, 
                    visualize: bool = True) -> bool:                
    """
    Command robot's movement in cartesian space.

    Parameters
    ----------
    waypoints : 1xN : obj : `list`
        waypoints (Pose objects) that the robot follows
    tp_heights : 1x2 : obj : `list` 
        heights of waypoints that the robot follow before grasping
    center : 3x1 : obj : `np.ndarray`
        array of potential gripper centers w.r.t the world frame
    direction : 3x1 : obj : `np.ndarray`
        array of potential gripper directions w.r.t the world frame
    top_down : bool
        whether execute top_down grasping or not
    visualize : bool
        whether visualize waypoints or not
        
    Returns
    -------
    success : bool
        whether the execution is successful or not
    waypoints : 1xN : obj : `list`
        waypoints (Pose objects) that the robot follows
    """
    if top_down_flag:
        waypoints, grasp_pose = top_down(waypoints, tp_heights, center, direction, 
                                                sub_aprvs)

    print("\n\n\n\ntop_down done!!!") #pannaga 
    print("Grasp Pose:\n\n", grasp_pose)
    return grasp_pose

def top_down(object_pose: Pose = None, tp_heights: list = None, 
                 center: np.ndarray = None, direction: np.ndarray = None,
                 sub_aprvs: list = None) -> list:
    """
    Plan top-down grasping by creating waypoints.

    Parameters
    ----------
    object_pose : obj : `Pose`
        pose of the object center w.r.t the world frame
    tp_heights : 1x2 : obj : `list` 
        heights of waypoints that the robot follow before grasping
    center : 3x1 : obj : `np.ndarray`
        array of potential gripper centers w.r.t the world frame
    direction : 3x1 : obj : `np.ndarray`
        array of potential gripper directions w.r.t the world frame
        
    Returns
    -------
    waypoints : 1xN : obj : `list`
        waypoints that the robot follows
    """
    waypoints = []
    wpose = copy.deepcopy(object_pose)

    # Put the gripper on top of the object center
    wpose.orientation.x = 0.0
    wpose.orientation.y = 1.0
    wpose.orientation.z = 0.0
    wpose.orientation.w = 0.0
    wpose.position.z += tp_heights[0]
    tdR = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
    norm_dir = direction / np.linalg.norm(direction)
    if sub_aprvs is None:
        # Aligh the gripper x-axis with the direction vector 
        # of a contact point pair 
        rot = rot_matrix(norm_dir, np.array([-1.,0.,0.])) @  tdR
        quat = R.from_matrix(rot).as_quat()
    else:
        # Execute a grasp configuration whose approach vector is closest 
        # to the z-axis
        dot_list, z_axis = [], [0, 0, 1]
        for aprv in sub_aprvs:
            dot_list.append(np.dot(aprv, z_axis))
        idx = dot_list.index(min(dot_list)) 
        selected_aprv = sub_aprvs[idx]
        print(selected_aprv)
        rot = Conversion().cpp2R(direction=norm_dir, aprv=selected_aprv)
        quat = R.from_matrix(rot).as_quat()

    temp = np.eye(4)
    temp[:3,:3] = rot
    temp[:3,3] = center
    res = temp @ np.array([0., 0., -tp_heights[1], 1.])
    
    wpose.orientation.x = quat[0]
    wpose.orientation.y = quat[1]
    wpose.orientation.z = quat[2]
    wpose.orientation.w = quat[3]
    wpose.position.x = res[0]
    wpose.position.y = res[1]
    wpose.position.z = res[2]
    waypoints.append(copy.deepcopy(wpose))

    # Move the gripper towards the grasping center
    res = temp @ np.array([0., 0., -0.195, 1.]) #how did these numbers come?

    wpose.position.x = res[0]
    wpose.position.y = res[1]
    wpose.position.z = res[2]
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.x = temp[0,3]
    wpose.position.y = temp[1,3]
    wpose.position.z = temp[2,3]
    print("###############  Waypoints  ############## \n", waypoints)
    print("================")
    print("##############   wpose  #############\n", wpose)
    return waypoints, wpose

def rot_matrix(axis_1: np.ndarray, axis_2: np.ndarray) -> np.ndarray:
    """ 
    Calculate a rotation matrix that aligns the axis_2 with the axis_1.
    
    Parameters
    ----------
    axis_1 : 1x3 : obj : `np.ndarray`
        unit direction vector 
    axis_2 : 1x3 : obj : `np.ndarray`
        unit direction vector

    Returns
    -------
    R : 3x3 :obj:`numpy.ndarray`
        rotation matrix
    """
    R = np.eye(3, dtype=np.float64)

    try:
        v = np.cross(axis_2, axis_1)
        c = np.dot(axis_2, axis_1)
        Vmat = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
        R = R + Vmat + Vmat @ Vmat / (1 + c)
    except:
        print('Vectors are in exactly opposite direction')

    return R


def main():
    single_grasp = True #can switch between True/False
    table_height = 0.70
    iter_sample = 1 
    success_prob = [] 


    pose = Pose()
    pose.position.x = 0
    pose.position.y = 0
    pose.position.z = 0.74   
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    print("Pose:", pose)



    path = "/home/bartonlab-user/workspace/src/kmr-iiwa-gripkit-cr-plus-l/test/4096_large_obj_04"
    centers, directions, aprvs, probs = grasp_gen(path, True, True, pose)
    refined, num_selected = avoid_collision(table_height, centers, directions, aprvs) 

    print(f'There are some infeasible grasp configurations in the provided set...')
    print(f'Number of refined configuration set: {num_selected} from {directions.shape[1]}')

    if num_selected < 1:
        raise Exception("There are no feasible grasps!!!") 
    if aprvs is not None:
        aprvs = [aprvs[i] for i in refined]
    if  single_grasp:
        if probs is not None:
            probs = [probs[i] for i in refined]
        
        res = execute(pose, centers[:,refined], directions[:,refined],  aprvs, probs, iter_sample)
        print(f"Result (grasp pose):\n {res}") #grasp pose
    else: 
        #for now I am not doing this 
        for idx in tqdm(refined):
            # Check the executing grasp configuration
            if idx:
                res = execute(centers[:,idx], directions[:,idx],  aprvs,iter_sample)
                if res >= 0:
                    success_prob.append(res)
                    print(f'Object: {obj}')
                    print(f'Probabilities: {success_prob}')
                    print(f'Mean: {np.mean(success_prob)}, STD: {np.std(success_prob)}')
                entire_list.append(res)
            else:
                entire_list.append(idx)






if __name__ == "__main__":
    main()
