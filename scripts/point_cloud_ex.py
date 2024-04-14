import cv2 
import matplotlib.pyplot as plt 

rgb = cv2.imread('/home/bartonlab-user/workspace/src/graspmixer_demo/rgb_raw_images/1708280820855648574.jpg')
depth = cv2.imread('/home/bartonlab-user/workspace/src/graspmixer_demo/depth_raw_images/1708280820850098014.png')
print(rgb)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# Display the RGB image
ax1.imshow(rgb)
ax1.axis('off')  # Turn off axis
ax1.set_title('RGB Image')

# Display the depth image
ax2.imshow(depth, cmap='gray')  # Use grayscale colormap for depth image
ax2.axis('off')  # Turn off axis
ax2.set_title('Depth Image')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
plt.show()