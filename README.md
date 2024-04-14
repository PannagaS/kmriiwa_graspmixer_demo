# graspmixer_demo




# *6D pose estimation and grasping experiments using KMR iiwa*

## Description
The goal of the project is to perform 6D pose estimation of [TLESS](http://cmp.felk.cvut.cz/t-less/) objects, and do grasping experiments. We use pose [Microsoft Azure DK](https://azure.microsoft.com/en-us/products/kinect-dk) RGB-D camera which is mounted on the kmr iiwa's link 7 (end effector), to perform pose estimation. Given the estimated 6D pose of the object, after a certain set of coordinate transformations, the robot is commanded to grasp the object. 

```
Current progress - the robot can grasp the object, and move to an instructed pose successfully.
Next steps (in progress) - implement a grasp planner to find an efficient grasp pose.
```

## Process flow
The pose estimation pipeline is as shown in the following figure. 
                                ![flow_chart (1)](https://github.com/PannagaS/ROB-590/assets/40464435/997cb707-6573-4d64-9270-8ab4098cbd96)

## Visuals

![grasping-ezgif com-video-to-gif-converter](https://github.com/PannagaS/ROB-590/assets/40464435/3d77487f-3fee-404a-9694-1bfaaaa83f0b)


## Code setup
```
📁 /graspmixer
├── 📄 CMakeLists.txt
├── 📄 LICENSE
├── 📄 README.md
├── 📄 large_obj_06.stl
├── 📁 launch
│   ├── 📄 kmriiwa_bringup_graspmixer.launch
│   ├── 📄 kmriiwa_bringup_graspmixer_cal.launch
│   ├── 📄 move_above_part.launch
│   ├── 📄 segmentation_testing.launch
│   ├── 📄 segmentation_testing_2.launch
│   ├── 📄 wrist_camera_graspmixer.launch
│   ├── 📄 wrist_camera_graspmixer_cal.launch
│   └── 📄 wristcam_demo.launch
├── 📁 output_models
│   ├── 📄 large_obj_01.stl
│   ├── 📄 large_obj_04.stl
│   ├── 📄 large_obj_06.stl
│   ├── 📄 large_obj_11.stl
│   ├── 📄 large_obj_14.stl
│   ├── 📄 large_obj_19.stl
│   ├── 📄 large_obj_25.stl
│   ├── 📄 small_obj_01.stl
│   ├── 📄 small_obj_04.stl
│   ├── 📄 small_obj_06.stl
│   ├── 📄 small_obj_11.stl
│   ├── 📄 small_obj_14.stl
│   ├── 📄 small_obj_19.stl
│   └── 📄 small_obj_25.stl
├── 📄 package.xml
├── 📄 pose_estimation_ros.py
├── 📁 scripts
│   ├── 📁 __pycache__
│   │   └── 📄 pose_estimation_globalRegistration_basic_class.cpython-38.pyc
│   ├── 📄 move_above_part.py
│   ├── 📄 point_cloud_ex.py
│   ├── 📄 pose_estimation.py
│   ├── 📁 remove these
│   │   ├── 📄 camera_info_modifier.py
│   │   ├── 📄 capture_img.py
│   │   ├── 📄 cv2_test.py
│   │   ├── 📄 dummy.txt
│   │   ├── 📄 extractHSVrange.py
│   │   ├── 📄 grey_box.png
│   │   ├── 📄 load_bag.py
│   │   ├── 📄 pose_est_pc_topic.py
│   │   ├── 📄 pose_estimation_globalRegistration.py
│   │   ├── 📄 pose_estimation_globalRegistration_basic.py
│   │   ├── 📄 pose_estimation_globalRegistration_basic_SDF.py
│   │   ├── 📄 pose_estimation_ros.py
│   │   ├── 📄 pose_estimation_server.py
│   │   ├── 📄 topic_2_image.py
│   │   ├── 📄 view_images.py
│   │   ├── 📄 view_pc.py
│   │   └── 📄 view_pc_class.py
│   └── 📄 usbcam_2_kinect_cal_math.py
├── 📁 src
│   └── 📄 manual_calibration_pub.py
└── 📁 temp
    └── 📄 mesh.stl

 ```

`launch/` contains all the necessary launch files to spin relevant nodes and topics.

`output_models/` contains CAD models of TLESS objects (big and small)

`scripts/` contains all the relevant codes of the project, important ones are     **pose_estimation_globalRegistration_basic_class.py** and **move_above_part.py**

`src/` contains the manual_calibration_pub.py that is used to facilitate the hand-eye calibration procedure




# Initial setup

## Hardware setup
Turn the robot on and wait for the pendant to boot up. Once the robot is fully booted, please proceed to select `LLBR_iiwa_7_R800`in the **Robot** drop down. 

Make sure you have ros core running before you proceed with the next step. 

Next, we need to turn on ROS driver from the pendant. 
From the **Applications** tab, please select `ROSKmriiwaController`. Once this is selected, you need to press enable, & press and hold the <ins>play</ins> button on the pendant till you see the message 'all nodes connected to the ROS master' (for about 10 seconds). 


## Software setup
Make sure you have the necessary dependencies installed in local.
### Dependencies  
`Python`  
`OpenCv`  
`Open3d`  
`numpy`  
`scipy`  
`ROS-Noetic`  
`tf_conversions`  
`cv_bridge`  
kmr iiwa :)  

Prior to implementing the entire pipeline, ie., pose estimation and grasping, please perform a hand-eye calibration. Follow [this link](https://ros-planning.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html) for more info.

### Performing Hand-eye calibration 
We are doing eye-in-hand calibration. 

Open a new terminal and start ros. 
```
rosrun -p 30001
```

Open another terminal, and execute the following:
```
roslaunch graspmixer_demo kmriiwa_bringup_graspmixer_cal.launch
```
Make sure you do not have any static_transform_publisher in **wrist_camera_graspmixer_cal.launch**. This is important as we intend to find the transformation of the camera w.r.t end effector. 

Make sure to select the following in hand-eye calibration gui:  
<ins>In target tab </ins>      
`image Topic` : `usb_cam_kmriiwa_link`  
`CameraInfo Topic` : `usb_cam_kmriiwa_link_info`

<ins>In context tab </ins>  
`Sensor frame` : `usb_cam_kmriiwa_link`     
`Object frame` : `handeye_target`   
`End-effector frame` : `kmriiwa_tcp`    
`Robot base frame` : `kmriiwa_link_0`


<ins>In calibrate tab</ins>

Choose `kmriiwa_manipulator` as Planning Group.

Take around 12-15 samples and save the pose file in a local directory. Copy the `static_transform_publisher` information (quaternion) and paste it in **wrist_camera_graspmixer.py**. 

You can verify the hand-eye calibration by visualizing the frame usb_cam_kmriiwa_wrist in rviz's tf_tree. The frame should originate at exactly where RGB lens of the camera resides.
## 
Assuming a good and reliable hand-eye calibration, 
open a new terminal and change the directory to *graspmixer_demo/launch*
```
roscd graspmixer_demo/launch
```

Execute the following after changing the directory
```
roslaunch graspmixer_demo kmriiwa_bringup_graspmixer.launch 
```


**kmriiwa_bringup_graspmixer.launch** launches `rviz`. This is used for visualizing robot model, object in depth cloud, and tf frames.

## Running the end-to-end process
In a new terminal, execute the following: 
```
roscd graspmixer_demo/launch

roslaunch graspmixer_demo move_above_part.launch 
```
The `move_above_part.launch` launch file is actually responsible to run the pose estimation pipeline and the robot control. 

## Some notes to consider
When you launch any launch file (except for move_above_part.launch), you are required to enable the pendant and press and hold <ins>play</ins> for 5-10 seconds (monitor the terminals). 

When launching **move_above_part.launch**, you need to press and hold play on the pendant before launching the script (hold till the end of execution) or till the control reaches embed(). You can let go when you work with embed, but if are doing any robot manipulation from a piece of code, you need to press and hold play on the pendant till the end of execution. 

## Authors and acknowledgment
Pannaga Sudarshan, Tyler Toner

## License
For open source projects, say how it is licensed.

## Project status
Implementing grasp planner for determining an efficient grasp pose.
