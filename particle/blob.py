import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

from distance import *

# Try opencv2.simpleblobdetector first

# Read image
# img = cv2.imread("../images/particles1.jpg", 0)
img = cv2.imread("../images/40000.jpeg", 0)

# Denoising
img_denoise = cv2.fastNlMeansDenoising(img, None, 30, 7, 21) # Mean denoising
# img_denoise = cv2.medianBlur(img, 7) # Median filter

# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(img_denoise)
# plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 255

# Set edge gradient
# params.thresholdStep = 5

# Filter by Area.
params.filterByArea = True
params.minArea = 5
# params.maxArea = 100

# Filter by Color
# params.filterByColor = True
# params.blobColor = 255

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img_denoise)
keypoints_coords = np.array([keypoint.pt for keypoint in keypoints])
keypoints_radius = np.array([keypoints.size for keypoints in keypoints])

# distance calculation
distances = distance_calc(keypoints_coords)
plt.hist(np.ravel(distances))
plt.show()
# distance filter
n = np.shape(keypoints_coords)[0]
cutoff_distance = 60
distance_filter = (distances <= cutoff_distance)
index_matrix = distance_filter * np.tril(np.ones((n, n), dtype=int), -1)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img_denoise, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
fig, axes = plt.subplots(2, 2)

axes[0, 0].title.set_text('Original image')
axes[0, 0].imshow(img, cmap = 'gray')

axes[0, 1].title.set_text('Denoise')
axes[0, 1].imshow(img_denoise, cmap = 'gray')

axes[1, 0].title.set_text('Blob detection')
axes[1, 0].imshow(im_with_keypoints)

axes[1, 1].title.set_text('Distance analysis')
axes[1, 1].imshow(img_denoise, cmap = 'gray')
distance_vis(keypoints_coords, index_matrix=index_matrix, axes=axes[1, 1])
for i, coord in enumerate(keypoints_coords):
    axes[1, 1].plot(coord[0], coord[1], 'bo')

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