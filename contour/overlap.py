# plot the overlap of all the contour & curvature in 
# the given directory.

import argparse
import csv
import sys
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def walker(rootdir, pattern = re.compile('.*?')):
    ls_filename = []
    ls_subdirs = []
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if pattern.match(file):
                ls_filename.append(file)
                ls_subdirs.append(subdirs)
    return zip(ls_filename, ls_subdirs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("contour_dir", type=Path, 
                        help="Directory that contains the contours & curvatures")
    args = parser.parse_args()
    working_directory = args.contour_dir
    pattern=re.compile('.*?csv$')
    x_all = []
    y_all = []
    curvature_all = []
    for filename, subdir in walker(working_directory, pattern):
        print("Adding "+filename)
        with open(Path(subdir).joinpath(Path(filename))) as f:
            reader = csv.reader(f)
            contour_curvature = np.array(list(reader)).astype(np.float)
            x_all.append(contour_curvature[:, 0].tolist())
            y_all.append(contour_curvature[:, 1].tolist())
            curvature_all.append(contour_curvature[:, 2].tolist())
    try:
        curvature_max = max([max(cv) for cv in curvature_all])
        curvature_min = min([min(cv) for cv in curvature_all])
    except ValueError:
        sys.exit("Can't find contour & curvature data in given directory.")

    # Plot and color mapping
    colormap = plt.get_cmap('jet')
    color_norm = matplotlib.colors.Normalize(vmin=curvature_min, vmax=curvature_max)
    scalar_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap=colormap)

    for i, _ in enumerate(x_all):
        scalar_map.set_array(curvature_all[i])
        plt.scatter(x_all[i], y_all[i], c=scalar_map.to_rgba(curvature_all[i]), marker='.', s=10)
    plt.gca().invert_yaxis() 
    plt.axis('equal')
    plt.colorbar(scalar_map)
    plt.show()
