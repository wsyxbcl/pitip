import matplotlib.pyplot as plt
import cv2
import numpy as np


# Try opencv2.simpleblobdetector first

# Read image
img = cv2.imread("../images/particles1.jpg", 0)
# img = cv2.imread("../images/particles.jpg", 0)
# Denoising

img_denoise = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img_denoise)
# plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Set edge gradient
# params.thresholdStep = 5

# Filter by Area.
# params.filterByArea = True
# params.minArea = 10
# params.maxArea = 100

# Filter by Color
# params.filterByColor = True
# params.blobColor = 255

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img_denoise)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img_denoise, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
plt.subplot(131)
plt.imshow(img, cmap = 'gray')
plt.subplot(132)
plt.imshow(img_denoise, cmap = 'gray')
plt.subplot(133)
plt.imshow(im_with_keypoints)

plt.show()

# Try different threshold

# img = cv2.medianBlur(img,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()