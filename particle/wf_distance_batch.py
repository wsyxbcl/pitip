import cv2
import numpy as np

# Create a black image, a window
img = cv2.imread("../images/particles.jpg", 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# create trackbars for denoising
cv2.createTrackbar('NlMean', 'image', 0, 1, lambda x: None) # fastNlMeansDenoising
cv2.createTrackbar('h', 'image', 10, 100, lambda x: None)
cv2.createTrackbar('tws', 'image', 3, 49, lambda x: None) # 0.5(templateWindowSize-1)
cv2.createTrackbar('sws', 'image', 10, 49, lambda x: None) # 0.5(searchWindowSize-1)

cv2.createTrackbar('Median', 'image', 0, 1, lambda x: None) # medianBlur
cv2.createTrackbar('k', 'image', 2, 49, lambda x: None) # 0.5(ksize-1)

cv2.imshow('image', img)

# Choose denoising method
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    h_mean = cv2.getTrackbarPos('h', 'image')
    tsize_mean = 2 * cv2.getTrackbarPos('tws', 'image') + 1
    ssize_mean = 2 * cv2.getTrackbarPos('sws', 'image') + 1
    ksize_median = 2 * cv2.getTrackbarPos('k', 'image') + 1
    if cv2.getTrackbarPos('NlMean', 'image') and cv2.getTrackbarPos('Median', 'image'):
        img_denoised_mean = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
        img_denoised = cv2.medianBlur(img_denoised_mean, ksize_median)
    elif cv2.getTrackbarPos('Median', 'image'):
        img_denoised = cv2.medianBlur(img, ksize_median)
    elif cv2.getTrackbarPos('NlMean', 'image'):
        img_denoised = cv2.fastNlMeansDenoising(img, None, h_mean, tsize_mean, ssize_mean)
    else:
        img_denoised = img
    cv2.imshow('image', img_denoised)

cv2.destroyAllWindows()