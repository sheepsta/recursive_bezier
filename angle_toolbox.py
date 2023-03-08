import numpy as np
import math


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def theta_calc(vector, previous_vector):
    try:
        theta = angle_between(
            (-1, 0, 0), (vector[0] - previous_vector[0], 0, vector[2] - previous_vector[2]))
        if math.isnan(theta):
            return 0
        return theta
    except:
        return 0


def phi_calc(vector, previous_vector):
    phi = angle_between((0, 1, 0), (vector[0] - previous_vector[0],
                                    vector[1] - previous_vector[1], vector[2] - previous_vector[2]))
    if phi == math.pi:
        return 0
    if math.isnan(phi):
        return 0
    else:
        return phi
