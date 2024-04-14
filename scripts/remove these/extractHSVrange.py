import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

def update_hue_range(event):
    global lower_hue, upper_hue
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'HSV Range')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'HSV Range')
    print("H upper = ", upper_hue)
    print("H lower = ", lower_hue)

def update_saturation_range(event):
    global lower_saturation, upper_saturation
    lower_saturation = cv2.getTrackbarPos('Lower Saturation', 'HSV Range')
    upper_saturation = cv2.getTrackbarPos('Upper Saturation', 'HSV Range')
    print("S upper = ", upper_saturation)
    print("S lower = ", lower_saturation)

def update_value_range(event):
    global lower_value, upper_value
    lower_value = cv2.getTrackbarPos('Lower Value', 'HSV Range')
    upper_value = cv2.getTrackbarPos('Upper Value', 'HSV Range')
    print("V upper = ", upper_value)
    print("V lower = ", lower_value)

def apply_segmentation():
    global image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([lower_hue, lower_saturation, lower_value])
    upper_bound = np.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    #result = cv2.bitwise_and(image, image, mask=mask)
    rect_mask = np.zeros(image.shape[:2], np.uint8)
    rect_mask[208:image.shape[1], 356:image.shape[0]] = 0
    combined_mask = mask | rect_mask
    result = cv2.bitwise_and(image, image, mask = combined_mask)
    cv2.imshow('Segmented Image', result)

def main():
    global image, lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value, upper_value
    path = '/home/bartonlab-user/workspace/src/graspmixer_demo/black_color/black_color.png'
    path = "/home/bartonlab-user/workspace/src/graspmixer_demo/scripts/grey_box_1.png"
    image = cv2.imread(path)
    cv2.bitwise_not(image)  # Replace 'image.jpg' with your image path
    cv2.namedWindow('HSV Range')
    cv2.createTrackbar('Lower Hue', 'HSV Range', 0, 179, update_hue_range)
    cv2.createTrackbar('Upper Hue', 'HSV Range', 0, 179, update_hue_range)
    cv2.createTrackbar('Lower Saturation', 'HSV Range', 0, 255, update_saturation_range)
    cv2.createTrackbar('Upper Saturation', 'HSV Range', 0, 255, update_saturation_range)
    cv2.createTrackbar('Lower Value', 'HSV Range', 0, 255, update_value_range)
    cv2.createTrackbar('Upper Value', 'HSV Range', 0, 255, update_value_range)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Press 'Esc' key to exit
            break
        apply_segmentation()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    lower_hue = 0
    upper_hue = 179
    lower_saturation = 0
    upper_saturation = 255
    lower_value = 0
    upper_value = 255
    main()
