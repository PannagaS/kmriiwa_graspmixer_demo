import numpy as np
import matplotlib.pyplot as plt

import cv2

path = "/home/bartonlab-user/workspace/src/graspmixer_demo/scripts/segmented_depth_image.jpeg"
img = cv2.imread(path, 0)


x1,y1, x2, y2 = 180, 48, 546, 521
mask = np.zeros(img.shape[:2], dtype = np.uint8)
mask[y1:y2, x1:x2] = 255
masked_image = cv2.bitwise_and(img, img, mask = mask)
contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for i in contours:
    area = cv2.contourArea(i)
    m = cv2.moments(i)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    print(cx,cy)
    print(area)
cv2.drawContours(masked_image, contours, -1, (255,0,0),3)
plt.subplot(1,2,1)
plt.title('DepthMask')
plt.imshow(img)
plt.subplot(1,2,2)
plt.title('DepthMask_focused')
plt.imshow(masked_image)
plt.show()