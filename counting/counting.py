import matplotlib.pyplot as plt
import cv2
import numpy as np


# Try opencv simpleblobdetector first

# Read image
im = cv2.imread("../images/particles1.jpg", cv2.IMREAD_COLOR)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Set edge gradient
params.thresholdStep = 5

# Filter by Area.
params.filterByArea = True
params.minArea = 20

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
plt.subplot(121)
plt.imshow(im)
plt.subplot(122)
plt.imshow(im_with_keypoints)

plt.show()