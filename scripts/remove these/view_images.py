import rospy
import cv2 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import imutils
"""
def callback(data):
    br = CvBridge()
    current_frame = br.imgmsg_to_cv2(data)
    bgr_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    log = bgr_frame
    rospy.loginfo(log)
    cv2.imshow('kinect camera', bgr_frame)
    cv2.waitKey(1)

def receive_message():
    rospy.init_node('kinect_sub_py', anonymous=True)
    rospy.Subscriber('depth_to_rgb/hw_registered/image_rect_raw', Image, callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':   

    receive_message()
"""

def get_distance(frame):    

    img_height = frame.shape[0]
    img_width = frame.shape[1]
    x_center = int(img_width/2)    
    
    y_center = int(img_height/2)
    distance = float(frame[int(y_center)][int(x_center)])*100.0
    distance_arr = frame[y_center-10:y_center+11, x_center-10:x_center+11]
    distance_arr = distance_arr.flatten()
    median = np.median(distance_arr)
    rospy.loginfo('The median distance is : %s cm', median*100.0)
    rospy.loginfo('The distance of center pixel is : %s cm', distance)

    return distance

def draw_crosshair(frame, crosshair_size = 5, crosshair_color=(255,0,0),stroke=1):
    img_height = frame.shape[0]
    img_width = frame.shape[1]
    x_center = img_width/2
    y_center = img_height/2
    cv2.line(frame, (x_center-crosshair_size, y_center-crosshair_size), (x_center+crosshair_size, y_center+crosshair_size), crosshair_color, stroke)
    cv2.line(frame, (x_center+crosshair_size, y_center-crosshair_size), (x_center-crosshair_size, y_center+crosshair_size), crosshair_color, stroke)
    cv2.rectangle(frame, (x_center - crosshair_size, y_center-crosshair_size), (x_center+crosshair_size, y_center+crosshair_size), (255,0,0),1)

def callback(data):
    br = CvBridge()
    current_frame = br.imgmsg_to_cv2(data, desired_encoding='32FC1')
    current_frame = imutils.resize(current_frame, width=600)
    current_frame = cv2.GaussianBlur(current_frame, (5,5),0)
    n_bgr_img = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('img',n_bgr_img)
    
    #distance = get_distance(current_frame)
    distance = 10
    bgr_img = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
    draw_crosshair(bgr_img, crosshair_size=10, crosshair_color=(0,0,255),stroke =1 )

    if distance!=0.0:
        cv2.putText(bgr_img, str(distance) +' cm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA, False)
    else:
        cv2.putText(bgr_img, "Too close/far", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA, False)

    cv2.imshow('depth_cam', bgr_img)
    cv2.imshow('depth cam no filter', n_bgr_img)
    cv2.waitKey(1)

def receive_message():
    rospy.init_node('kinect_sub_py', anonymous=True)
    rospy.Subscriber('/depth_to_rgb/hw_registered/image_rect_raw', Image, callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__=='__main__':
    receive_message()
 
