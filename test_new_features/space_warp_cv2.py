import cv2
import numpy as np

# Load the image
img = cv2.imread("image.png")

# Define the source points
src_points = np.float32([[0,0],[0,100],[100,100],[100,0]])

# Define the destination points
dst_points = np.float32([[50,50],[0,150],[150,150],[100,50]])

# Calculate the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the transformation to the image
warped_img = cv2.warpPerspective(img, M, (200, 200))

# Save the warped image
cv2.imwrite("warped_image.jpg", warped_img)