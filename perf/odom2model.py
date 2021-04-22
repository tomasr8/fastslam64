import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

np.random.seed(0)

V = 0.05

control = np.zeros((201, 2), dtype=np.float64)
control[1:, 1] = V

np.save("control.npy", control)