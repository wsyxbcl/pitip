import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


# Try opencv simpleblobdetector first

# Read image
img = cv.imread("../images/40000.jpeg", 0)

# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 150
params.maxThreshold = 255

# Set edge gradient
params.thresholdStep = 5

# Filter by Area.
params.filterByArea = True
params.minArea = 20
# params.maxArea = 100

# Filter by Color
# params.filterByColor = True
# params.blobColor = 255

# Set up the detector with default parameters.
detector = cv.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.subplot(122)
plt.imshow(im_with_keypoints)

plt.show()

# Try different threshold

# img = cv.medianBlur(img,5)
# ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()