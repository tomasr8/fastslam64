import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def coords_to_rb(measurement, theta):
    r = np.linalg.norm(measurement)
    b = np.arctan2(measurement[1], measurement[0]) - theta
    b = wrap_angle(b)

    return r, b


for i in range(1, 6):
    print(i)
    ground = np.load(f"npy/ground_{i}_50hz.npy")
    measurements_xy = np.load(f"npy/measurements_xy_{i}_50hz.npy")

    measurements_rb = np.zeros_like(measurements_xy)

    for j, m in enumerate(measurements_xy):
        t = m[0]
        g = ground[ground[:, 0] == t][0]

        # print(g)

        theta = g[3]
        r, b = coords_to_rb(m[2:], theta)

        measurements_rb[j] = [m[0], m[1], r, b]


    np.save(f"npy_fixed/measurements_rb_{i}_50hz.npy", measurements_rb)