#!/usr/bin/env python3

import argparse
import os
import sys
import csv

import numpy as np
import time
from imageio import imread
from scipy.ndimage import convolve
import matplotlib
import matplotlib.pyplot as plt
import morphsnakes as ms
from pathlib import Path

from curvature import cal_curvature


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.
    
    Args
        background: (M, N) array
            Image to be plotted as the background of the visual evolution.
        fig: matplotlib.figure.Figure
            Figure where results will be drawn. If not given, a new figure
            will be created.
    
    Returns
        callback: Python function
            A function that receives a levelset and updates the current plot
            accordingly. This can be passed as the `iter_callback` argument of
            `morphological_geodesic_active_contour` and
            `morphological_chan_vese`.
    
    """
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('equal')
    ax.imshow(background, cmap=plt.cm.gray)

    def callback(levelset):
        global img_contour
        # print(levelset)
  
        if ax.collections:
            del ax.collections[0]
        img_contour = ax.contour(levelset, [0.5], colors='r').allsegs[0][0]
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

def rgb2gray(img):
    """
    Convert a RGB image to gray scale.
    """
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def smooth(x, kernel_size, wrap=1):
    """
    Convolution on 1d array to smooth the line

    Args
        x: 1d array
        wraps: 0 or 1
            When wrap == 1, wrap mode is used to do a circular convolution
        kernel_size: integer
            Kernel(or filter) size in convolution
    """
    kernel = np.ones(kernel_size)/kernel_size
    # x_smooth = np.convolve(x, kernel, mode='valid') # edge info lost
    if wrap:
        x_smooth = convolve(x, kernel, mode='wrap')
    else:
        x_smooth = np.convolve(x, kernel, mode='valid')
    return x_smooth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path, help="Path to your aimed image")
    args = parser.parse_args()
    PATH_IMG_TEM = args.image_dir

    ################ Parameters for contour detection ################
    # Morphsnake Algorithm
    # size of the circle level_set (radius/min(img.shape))
    circle_ratio = 0.49 # 0 ~ 0.5
    # time of iterations
    iterations = 300
    # weight parameters
        # If 'lambda1' is larger than 'lambda2', the outer
        # region will contain a larger range of values than
        # the inner region.(Vice versa)
    lambda1 = 1
    lambda2 = 1 
    smoothing = 1 # int, reasonably 1~4

    # Convolve & Calculate Curvature
    # whether the contour is cyclic
    cyclic_contour = 1 # 0 or 1
    # kernel_size(in smooth) and interval(in curvature calculation) 
    # are set later in terminal
    #################################################################

    PATH_CONTOUR_CSV = Path('./output_contour_cv').joinpath(PATH_IMG_TEM.stem+'.csv')
    if not PATH_CONTOUR_CSV.parent.exists():
        PATH_CONTOUR_CSV.parent.mkdir()
    # Global variable to store coordinates of contour points
    global img_contour
    img_contour = []

    # Load the image.
    try:
        imgcolor = imread(str(PATH_IMG_TEM))/255.0
    except FileNotFoundError:
        print("No such file, please check.")
        sys.exit()
    
    if imgcolor.ndim != 2:
        img = rgb2gray(imgcolor)
    else:
        img = imgcolor
    
    # Initialization of the level-set.
    init_ls = ms.circle_level_set(img.shape, radius=min(img.shape) * circle_ratio)
    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese
    ms.morphological_chan_vese(img, iterations=iterations,
                               init_level_set=init_ls,
                               smoothing=smoothing, lambda1=lambda1, lambda2=lambda2,
                               iter_callback=callback)

    # ms.morphological_chan_vese(img, iterations=iterations,
    #                            smoothing=3, lambda1=lambda1, lambda2=lambda2,
    #                            iter_callback=callback)


    # Try kernel_size(in smooth) and interval(in curvature calculation)
    while True:
        # Contour Smooth
        try:
            if input("Command(Press enter to continue, q for quit): ").lower() == 'q':
                break
            kernel_size = int(input("Kernel size\n> ")) 
            x = img_contour[:, 0]
            y = img_contour[:, 1]
            x_smooth = smooth(x, kernel_size=kernel_size, wrap=cyclic_contour)
            y_smooth = smooth(y, kernel_size=kernel_size, wrap=cyclic_contour)
            # Curvature calculation
            interval = int(input("Point interval\n> "))

            curvature = cal_curvature(x_smooth, y_smooth, cyc=cyclic_contour, interval=interval)

            with open(PATH_CONTOUR_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(zip(x_smooth, y_smooth, curvature))

            curvature_sorted = sorted(curvature)
            # Stupid idea...
            # curvature_min = curvature_sorted[int(0.2*len(curvature))]
            # curvature_max = curvature_sorted[-int(0.2*len(curvature))]   
            curvature_min = curvature_sorted[0]
            curvature_max = curvature_sorted[-1]

            # Plot and color mapping
            colormap = plt.get_cmap('jet')
            color_norm = matplotlib.colors.Normalize(vmin=curvature_min, vmax=curvature_max)
            scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=colormap)
            scalar_map.set_array(curvature)

            imgcolor = imread(str(PATH_IMG_TEM))/255.0
            img = rgb2gray(imgcolor)
            
            plt.close()
            plt.subplot(111)
            plt.axis('equal')
            plt.imshow(img, cmap=plt.cm.gray)
            plt.scatter(x_smooth, y_smooth, c=scalar_map.to_rgba(curvature), marker='.', s=10)
            # ax.plot(x, y, c=scalar_map.to_rgba(curvature))
            plt.colorbar(scalar_map)

            # plt.subplot(122)
            # plt.scatter(x_smooth, y_smooth, c=scalar_map.to_rgba(curvature), marker='.', s=10)
            # plt.gca().invert_yaxis() # flip y axis
            # ax.plot(x, y, c=scalar_map.to_rgba(curvature))
            # plt.colorbar(scalar_map)
            plt.show()
         
        except ValueError:
            print("Error: An integer is expected.")
            continue
        except EOFError:
            break
    print("Bye")
