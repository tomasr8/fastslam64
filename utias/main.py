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

        # if ms.shape[0] > 0:
        #     print("ground time:", t, x, y, theta, "z", ms)
        i += ms.shape[0]

        for (_, b, rng, bearing) in ms:
            if b in barcodes:
                lm_x, lm_y = to_coords(rng, bearing, theta)
                new.append([t, b, lm_x, lm_y])

    print(i)

    return np.array(new)


def to_coords(range, bearing, theta):
    return [range * np.cos(bearing + theta), range * np.sin(bearing + theta)]


def get_new_pos(old_pos, odom, dt):
    theta = old_pos[2] + dt * odom[1]
    dist = dt * odom[0]
    x = old_pos[0] + dist * np.cos(theta)
    y = old_pos[1] + dist * np.sin(theta)

    return [x, y, theta]


landmarks = np.load("landmarks.npy")
for i in range(1, 6):
    ground = genfromtxt(f"csv/ground_{i}_50hz.csv", delimiter=",")
    np.save(f"npy/ground_{i}_50hz.npy", ground)

    odom = genfromtxt(f"csv/odom_{i}_50hz.csv", delimiter=",")
    np.save(f"npy/odom_{i}_50hz.npy", odom)

    measurements = genfromtxt(f"csv/measurements_{i}_50hz.csv", delimiter=",")
    np.save(f"npy/measurements_{i}_50hz.npy", measurements)

    new_measurements = rb_to_coords(ground, measurements, landmarks[:, 0])
    print(measurements.shape)
    print(new_measurements.shape)
    np.save(f"npy/measurements_xy_{i}_50hz.npy", new_measurements)


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
