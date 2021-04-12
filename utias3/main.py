import math
import numpy as np
from numpy.testing._private.utils import measure
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from numpy import genfromtxt


def filter_measurements(measurements, barcodes):
    measurements = [m for m in measurements if m[1] in barcodes]
    return measurements


# barcodes = genfromtxt("barcodes.csv", delimiter=",")
# landmarks = genfromtxt("landmarks.csv", delimiter=",")
# np.save("barcodes.npy", barcodes)
# np.save("landmarks.npy", landmarks)

barcodes = np.load("barcodes.npy")
landmarks = np.load("landmarks.npy")

landmark_barcodes = barcodes[5:, 1]


for i in range(1, 6):
    print(i)
    ground = genfromtxt(f"csv/ground_{i}_50hz.csv", delimiter=",")
    np.save(f"npy/ground_{i}_50hz.npy", ground)

    odom = genfromtxt(f"csv/odom_{i}_50hz.csv", delimiter=",")
    np.save(f"npy/odom_{i}_50hz.npy", odom)

    measurements = genfromtxt(f"csv/measurements_{i}_50hz.csv", delimiter=",")
    measurements = filter_measurements(measurements, landmark_barcodes)
    np.save(f"npy/measurements_{i}_50hz.npy", measurements)


# T = 100000.0
# ground = np.load("ground_10hz.npy")
# odom = np.load("odom_10hz.npy")
# # measurements = np.load("measurements.npy")
# measurements = np.load("measurements_10hz.npy")
# landmarks = np.load("landmarks.npy")

# ground = ground[ground[:, 0] < T]
# odom = odom[odom[:, 0] < T]
# measurements = measurements[measurements[:, 0] < T]


# new_measurements = rb_to_coords(ground, measurements, landmarks[:, 0])
# print(measurements.shape)
# print(new_measurements.shape)
# np.save("measurements_xy_10hz.npy", new_measurements)


# fig, ax = plt.subplots()
# ax.scatter(landmarks[:, 1], landmarks[:, 2], c="red")
# pos = ground[600, 1:]

# for i, (g, o) in enumerate(zip(ground[600:], odom[600:])):

    # if i % 100 == 0:
    #     ax.scatter(g[1], g[2], c="green")
    #     ax.scatter(pos[0], pos[1], c="orange")
    #     plt.pause(0.1)

    # t = g[0]
    # m = measurements[measurements[:, 0] == t]
    # if m.size > 0:
    #     print(m)
    #     ax.scatter(g[1], g[2], c="green")
    #     # ax.scatter(pos[0], pos[1], c="orange")
    #     plt.pause(0.01)

    #     lms = []
    #     for l in m:
    #         if l[1] in landmarks[:, 0]:
    #             lms.append(to_coords(l[2], l[3], g[3]))



    #     for lm in lms:
    #         ax.scatter(g[1] + lm[0], g[2] + lm[1], c="purple")


    # pos = get_new_pos(pos, o[1:], 0.02)


# barcodes = genfromtxt("barcodes.csv", delimiter=",")
# landmarks = np.load("landmarks.npy")

# print(barcodes)
# barcodes = { int(n): b for (n, b) in barcodes }
# print(barcodes)

# for l in landmarks:
#     n = int(l[0])
#     l[0] = barcodes[n]

# print(landmarks)
# np.save("landmarks.npy", landmarks[:, [0,1,2]])

# ground = genfromtxt("ground_10hz.csv", delimiter=",")
# np.save("ground_10hz.npy", ground)

# odom = genfromtxt("odom_10hz.csv", delimiter=",")
# np.save("odom_10hz.npy", odom)

# measurements = genfromtxt("measurements_10hz.csv", delimiter=",")
# np.save("measurements_10hz.npy", measurements)
