import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

from distance import *

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

def particle_blob_detection(img, min_area=5):
    """
    Using cv2.SimpleBlobDector for blob detection
    """
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 255

    # Set edge gradient
    params.thresholdStep = 5

    # Filter by Area.
    params.filterByArea = True
    params.minArea = min_area
    # params.maxArea = 100

    # Filter by Color
    # params.filterByColor = True
    # params.blobColor = 255

    # Set up the detector with parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img_denoise)
    return keypoints


if __name__ == '__main__':
    # Read image
    img = cv2.imread("../images/particles.jpg", 0)

    # Denoising
    img_denoise = cv2.fastNlMeansDenoising(img, None, 30, 7, 21) # Mean denoising   
    # img_denoise = cv2.medianBlur(img, 7) # Median filter

    # Blob detection
    keypoints = particle_blob_detection(img_denoise)
    keypoints_coords = np.array([keypoint.pt for keypoint in keypoints])
    keypoints_radius = np.array([keypoints.size for keypoints in keypoints])

    # Distance calculation
    distances = distance_calc(keypoints_coords)
    fig1, ax = plt.subplots()
    plt.hist(np.ravel([distance for distance in np.ravel(distances) if distance != 0]), bins=30, density=1)
    plt.title('Histogram of distance')
    plt.xlabel('Distances')
    plt.ylabel('Probability')
    plt.grid(True)
    fig1.tight_layout()

    # distance filter
    n = np.shape(keypoints_coords)[0]
    # print(n)
    # print(np.shape(distances))
    # print(np.shape([distance for distance in np.ravel(distances) if distance != 0]))
    cutoff_distance = 35
    distance_filter = (distances <= cutoff_distance)
    index_matrix = distance_filter * np.tril(np.ones((n, n), dtype=int), -1)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img_denoise, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints

    fig2 = plt.figure(tight_layout=True)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    # fig, axes = plt.subplots(3, 2)
    ax1 = plt.subplot(gs[0, 0])  
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    # ax5 = plt.subplot(gs[2, 0])
    
    ax1.title.set_text('Original image')
    ax1.imshow(img, cmap = 'gray')

    ax2.title.set_text('Denoise')
    ax2.imshow(img_denoise, cmap = 'gray')

    ax3.title.set_text('Blob detection')
    ax3.imshow(im_with_keypoints)

    ax4.title.set_text('Distance analysis (cutoff = {})'.format(cutoff_distance))
    ax4.imshow(img_denoise, cmap = 'gray')
    distance_vis(keypoints_coords, index_matrix=index_matrix, axes=ax4)
    for i, coord in enumerate(keypoints_coords):
        ax4.plot(coord[0], coord[1], 'bo')

    fig2.tight_layout()
    plt.show()