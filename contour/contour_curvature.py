import os

import numpy as np
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

def smooth(x, kernel_size):
    """
    circular convolution(wrap mode) on 1d array to smooth the line
    """
    kernel = np.ones(kernel_size)/kernel_size
    # x_smooth = np.convolve(x, kernel, mode='valid') # edge info lost
    x_smooth = convolve(x, kernel, mode='wrap')
    return x_smooth

def example_tem():
    # Load the image.
    imgcolor = imread(str(PATH_IMG_TEM))/255.0
    img = rgb2gray(imgcolor)
    
    # Initialization of the level-set.
    init_ls = ms.circle_level_set(img.shape, radius=min(img.shape) * 3.3 / 8.0)
    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese
    ms.morphological_chan_vese(img, iterations=100,
                               init_level_set=init_ls,
                               smoothing=3, lambda1=1, lambda2=5,
                               iter_callback=callback)


if __name__ == '__main__':

    PATH_IMG_TEM = Path('../images/8.tif')

    # Global variable to store coordinates of contour points
    global img_contour
    img_contour = []

    example_tem()
    # plt.show()
    
    # Calculate the curvature
    x = img_contour[:, 0]
    y = img_contour[:, 1]
    x_smooth = smooth(x, 30)
    y_smooth = smooth(y, 30)

    curvature = cal_curvature(x_smooth, y_smooth, interval=30)
    curvature_sorted = sorted(curvature)
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
    plt.subplot(121)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.scatter(x_smooth, y_smooth, c=scalar_map.to_rgba(curvature), marker='.', s=10)
    # ax.plot(x, y, c=scalar_map.to_rgba(curvature))
    plt.colorbar(scalar_map)

    plt.subplot(122)
    plt.scatter(x_smooth, y_smooth, c=scalar_map.to_rgba(curvature), marker='.', s=10)
    plt.gca().invert_yaxis() # flip y axis
    # ax.plot(x, y, c=scalar_map.to_rgba(curvature))
    plt.colorbar(scalar_map)
    plt.show()
