import math
import numpy as np

from distance import distance_calc

distances = distance_calc(particles)

cutoff_distance =
distance_filter_matrix = (distances <= cutoff_distance)
np.fill_diagonal(distance_filter_matrix, 0)

phi_6_j = []
for j_idc, distance_filter in enumerate(distance_filter_matrix):
    k_idx = np.where(distance_filter == 1)[0]
    n_k = k_idx.size 
    bond_jk = particles[k_idx] - particles[j_idc] # n_k * 2 matrix
    theta_jk = np.arctan2(bond_jk[:, 1], bond_jk[:, 0])
    phi_6_j.append(np.average(np.exp(6j*theta_jk)))
psi_6 = np.average(phi_6_j)