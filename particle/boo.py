import math
import numpy as np

from distance import distance_calc

distances = distance_calc(particles)

cutoff_distance =
distance_filters = (distances <= cutoff_distance)
for j_idx, distance_filter in distance_filters:
    k_idx = np.where(distance_filter == 1)[0]
    n_k = k.size
    bond_jk = particles[k] - particles[j]
    theta_jk = np.arctan2(bond_jk[1], bond_jk[0]) #TODO could be vectorized
