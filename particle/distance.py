import numpy as np
import matplotlib.pyplot as plt

_no_value = object()

def distance_calc(points):
    """
    Calculate the distance between every 2 points in given array
    (numpy array that contains n coordinates of points)
    and return a matrix containiing mutual distances 
    """
    n = np.shape(points)[0] # number of points
    distance = np.empty((n, n))
    for i, point in enumerate(points):
        distance[:, i] = np.linalg.norm((points - point), axis=1)
    return distance

def distance_vis(points, index_matrix=_no_value, axes=_no_value):
    """
    Visualize distances between points by connecting them with line.
    An Index Matrix is given to filter out unwanted distances.
    """
    n = np.shape(points)[0]
    # connect every points if no index matrix is given
    if index_matrix is _no_value:
        index_matrix = np.tril(np.ones((n, n), dtype=int), -1)
    targets = np.where(index_matrix == 1)
    if axes is _no_value:
        for i, first_idx in enumerate(targets[0]):
            plt.plot((points[first_idx][0], points[targets[1][i]][0]), 
                     (points[first_idx][1], points[targets[1][i]][1]), '-r')
    else:
        for i, first_idx in enumerate(targets[0]):
            axes.plot((points[first_idx][0], points[targets[1][i]][0]), 
                      (points[first_idx][1], points[targets[1][i]][1]), '-r')        
    # plt.show()


if __name__ == '__main__':
    a = np.array([[1, 1], [2, 1], [3, 2], [2, 4], [0, 2]])
    n = np.shape(a)[0]
    distances = distance_calc(a)
    plt.plot(a[:, 0], a[:, 1], 'o')
    distance_vis(a)
    plt.show()