#!/usr/bin/env python3

import cv2
import numpy as np

def scale_contour(contours, scale):
    contour_moments = []
    for contour in contours:
        contour_moments.append(cv2.moments(contour))
    # Mass centers
    mass_centers = []
    contours_scaled = []
    for i, m in enumerate(contour_moments):
        mass_center = (m['m10'] / (m['m00'] + 1e-5), m['m01'] / (m['m00'] + 1e-5))
        contour_normed = contours[i] - mass_center
        contour_scaled = (contour_normed * scale) + mass_center
        contours_scaled.append(contour_scaled.astype(np.int32))
    return contours_scaled

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def erode_mask(mask, kernel_size=3, num_iterations=5):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    kernel[[0, 0, -1, -1], [0, -1, 0, -1]] = 0 # for cross-like kernel
    eroded = cv2.erode(mask, kernel, iterations=num_iterations)
    return eroded

if __name__ == '__main__':
    im = cv2.imread("../images/sinyuan.png", cv2.IMREAD_UNCHANGED)
    alpha_channel = im[:,:,3]
    contours, hierarchy = cv2.findContours(alpha_channel, cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    mask = mask_from_contours(im, contours)
    mask_eroded = erode_mask(mask, kernel_size=3, num_iterations=8)
    contour_removed = cv2.bitwise_and(im, im, mask=mask_eroded)
    cv2.imwrite('../images/contour_removal.png', contour_removed)