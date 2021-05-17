import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import matplotlib

def get_length(odom):
    s = 0

    for i in range(len(odom) - 1):
        s += np.linalg.norm(odom[i, :2] - odom[i+1, :2])

    return s

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def lidar_to_global(cones, x, y, heading):
    # print(x, y, np.rad2deg(heading))
    cones = np.array(cones).copy()
    cones[:, 0] += 1.2

    R = np.float32([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])


    cones = np.matmul(R, cones.T).T
    cones[:, 0] += x
    cones[:, 1] += y

    return cones

def process(data, N):
    odom = []
    measurements = []

    for i in range(80, N):
        x, y, theta = data[i]["odom"]
        theta = pi_2_pi(theta)
        odom.append([x, y, theta])

        cones = data[i]["measurements"]
        cones = lidar_to_global(cones, x, y, theta)

        for cone in cones:
            # cone[0] += x + 1.5
            # cone[1] += y

            measurements.append(cone)


    measurements = np.array(measurements)

    odom = np.array(odom)
    # print(odom)
    odom[:, [0, 1, 2]] = odom[:, [1, 0, 2]]
    odom[:, 1] +=2
    odom[:, 0] *= -1
    odom[:, 2] += np.pi/2
    odom[:, 2] = pi_2_pi(odom[:, 2])

    measurements[:, [0, 1]] = measurements[:, [1, 0]]
    measurements[:, 1] +=2
    measurements[:, 0] *= -1

    return odom, measurements


def process_compare(data, A, B):
    true_odom = []
    x_odom = []
    true_measurements = []
    x_measurements = []

    for i in range(A, B):
        x, y, theta = data[i]["true_odom"]
        theta = pi_2_pi(theta)
        true_odom.append([x, y, theta])

        cones = np.array(data[i]["measurements"])
        cones = lidar_to_global(cones, x, y, theta)

        for cone in cones:
            true_measurements.append(cone)

        x, y, theta = data[i]["xavier_odom"]
        theta = pi_2_pi(theta)
        x += 0.4*np.cos(theta)
        y += 0.4*np.sin(theta)
        x_odom.append([x, y, theta])

        cones = np.array(data[i]["measurements"])
        cones = lidar_to_global(cones, x, y, theta)

        for cone in cones:
            x_measurements.append(cone)


    true_measurements = np.array(true_measurements)
    x_measurements = np.array(x_measurements)

    true_odom = np.array(true_odom)
    x_odom = np.array(x_odom)
    # print(odom)
    true_odom[:, [0, 1, 2]] = true_odom[:, [1, 0, 2]]
    true_odom[:, 1] +=2
    true_odom[:, 0] *= -1
    true_odom[:, 2] += np.pi/2
    true_odom[:, 2] = pi_2_pi(true_odom[:, 2])

    x_odom[:, [0, 1, 2]] = x_odom[:, [1, 0, 2]]
    x_odom[:, 1] +=2
    x_odom[:, 0] *= -1
    x_odom[:, 2] += np.pi/2
    x_odom[:, 2] = pi_2_pi(x_odom[:, 2])

    true_measurements[:, [0, 1]] = true_measurements[:, [1, 0]]
    true_measurements[:, 1] +=2
    true_measurements[:, 0] *= -1

    x_measurements[:, [0, 1]] = x_measurements[:, [1, 0]]
    x_measurements[:, 1] +=2
    x_measurements[:, 0] *= -1

    return true_odom, x_odom, true_measurements, x_measurements


# with open("odom_measurements_relative_true.json") as f:
#     data = json.load(f)


# N = len(data)
# true_odom, true_measurements = process(data, N)

# plt.scatter(true_odom[:, 0], true_odom[:, 1], c="green", s=5)
# plt.scatter(true_measurements[:, 0], true_measurements[:, 1], c="purple", s=3)
# track = np.load("../track.npy")
# plt.scatter(track[:, 0], track[:, 1], c="blue", s=4)

# with open("odom_measurements_relative.json") as f:
#     data = json.load(f)

# N = len(data)
# x_odom, x_measurements = process(data, N)


# plt.scatter(x_odom[:, 0], x_odom[:, 1], c="green", s=5)
# plt.scatter(x_measurements[:, 0], x_measurements[:, 1], c="purple", s=3)
# track = np.load("../track.npy")
# plt.scatter(track[:, 0], track[:, 1], c="blue", s=4)

# plt.plot(true_odom[:, 2], c="green")


# plt.plot(np.concatenate((np.zeros(37), x_odom[:, 2])), c="orange")


with open("odom_measurements_compare.json") as f:
    data = json.load(f)

A = 80
B = 90
A = 0
B = 880
true_odom, x_odom, true_measurements, x_measurements = process_compare(data, A, B)


EXPORT = False

if EXPORT:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


# fig, ax = plt.subplots()
# fig.set_size_inches(w=5.02, h=5.02)
# fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.99)

# plt.plot(true_odom[:, 2], c="green")
# plt.plot(x_odom[:, 2], c="orange")

# plt.plot(true_odom[:, 0], c="green")
# plt.plot(x_odom[:, 0])

# plt.plot(true_odom[:, 1], c="green")
# plt.plot(x_odom[:, 1])

# for (x, y, theta) in true_odom:
#     # plt.quiver([x], [y], [x+0.1*np.cos(theta)], [y+0.1*np.sin(theta)], width=0.002)
#     plt.plot([x, x+0.2*np.cos(theta)], [y, y+0.2*np.sin(theta)], c="green")
#     # plt.plot([x, x+15*np.cos(theta)], [y, y+15*np.sin(theta)], c="green", linewidth=0.1)

plt.plot(true_odom[:, 0], true_odom[:, 1], linestyle='dashed', c="green", label="Robot path")
# marker='o', c="green", markersize=4, markevery=5

# for (x, y, theta) in x_odom:
#     plt.plot([x, x+0.2*np.cos(theta)], [y, y+0.2*np.sin(theta)], c="orange")
#     # plt.plot([x, x+15*np.cos(theta)], [y, y+15*np.sin(theta)], c="orange", linewidth=0.1)

# plt.scatter(x_odom[:, 0], x_odom[:, 1], c="orange", s=5)


# true_measurements = true_measurements[true_measurements[:, 1] < 18]
# x_measurements = x_measurements[x_measurements[:, 1] < 18]


# plt.scatter(true_measurements[:, 0], true_measurements[:, 1], c="orange", s=3)
plt.scatter(true_measurements[:, 0], true_measurements[:, 1], c="green", s=3, label="Measurements")

plt.scatter(x_measurements[:, 0], x_measurements[:, 1], c="orange", s=3, label="Measurements")

track = np.load("../track.npy")
# track = track[(track[:, 0] > -5) & (track[:, 0] < 5)]
# track = track[(track[:, 1] > 0) & (track[:, 1] < 11)]
plt.scatter(track[:, 0], track[:, 1], marker=(7, 1, 0), c="blue", s=35, label="Landmarks")

# ax.set_xticks([])
# ax.set_yticks([])


# plt.xlim(-124, -97)
# plt.ylim(9, 21.5)


# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0),
#           fancybox=False, shadow=False, ncol=3, columnspacing=0.5)



print(get_length(true_odom))
# plt.axis("equal")

if EXPORT:
    plt.savefig('fsonline_histogram.pgf')
else:
    plt.show()