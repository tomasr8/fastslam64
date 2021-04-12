import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from numpy import genfromtxt


def rb_to_coords(ground, measurements, barcodes):
    new = []

    i = 0
    for t, x, y, theta in ground:
        ms = measurements[measurements[:, 0] == t]

        if ms.size != 0:
            print(t, ms)
            print("===")
        # if ms.shape[0] > 0:
        #     print("ground time:", t, x, y, theta, "z", ms)
        i += ms.shape[0]

        for (_, b, rng, bearing) in ms:
            if int(b) in barcodes:
                lm_x, lm_y = to_coords(rng, bearing, theta)
                new.append([t, b, lm_x, lm_y])

        if t > 1000:
            break

    print(i)
    print(len(new))

    return np.array(new)


def to_coords(range, bearing, theta):
    return [range * np.cos(bearing + theta), range * np.sin(bearing + theta)]


landmarks = np.load("landmarks.npy")
ground = genfromtxt(f"csv/ground_1_50hz.csv", delimiter=",")
measurements = genfromtxt(f"csv/measurements_1_50hz.csv", delimiter=",")

barcodes = list(map(int, landmarks[:, 0]))
print(barcodes)

new_measurements = rb_to_coords(ground, measurements, barcodes)