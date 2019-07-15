import math
import numpy as np
import matplotlib.pyplot as plt

from distance import distance_calc

def phi_6(points, cutoff_distance):
    """
    Input
        points: n*2 numpy array that contains n coordinates of points
        cutoff_distance: two points are defined as nearest neighbors if
                         the separation is within the cutoff_distance.
                         (for now, unit: pixel)
    Return
        phi_6_j: np array, n_phi_6 of every point j

    """
    distances = distance_calc(points)
    distance_filter_matrix = (distances <= cutoff_distance)
    np.fill_diagonal(distance_filter_matrix, 0)

    phi_6_j = []
    for j_idc, distance_filter in enumerate(distance_filter_matrix):
        k_idx = np.where(distance_filter == 1)[0]
        n_k = k_idx.size 
        if n_k == 0 or n_k == 1:
            phi_6_j.append(0)
        else:
            bond_jk = points[k_idx] - points[j_idc] # n_k * 2 matrix
            theta_jk = np.arctan2(bond_jk[:, 1], bond_jk[:, 0])
            phi_6_j.append(np.average(np.exp(6j*theta_jk)))
    phi_6_j = np.absolute(phi_6_j)
    psi_6 = np.average(phi_6_j)
    return phi_6_j

if __name__ == '__main__':
    # Test case
    particles = np.random.rand(30, 2)
    # plt.scatter(particles[:, 0], particles[:, 1])
    # plt.show()
    phi_6_j = phi_6(particles, cutoff_distance=0.1)