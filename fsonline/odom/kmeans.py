import math
import numpy as np
from numpy.testing._private.utils import measure
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json

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


def process_compare(data):
    odom = []
    measurements = []

    for i in range(len(data)):
        x, y, theta = data[i]["true_odom"]
        theta = pi_2_pi(theta)
        odom.append([x, y, theta])

        cones = data[i]["measurements"]
        cones = lidar_to_global(cones, x, y, theta)

        # measurements.append([])
        for cone in cones:
            cone[0], cone[1] = cone[1], cone[0]
            cone[1] += 2
            cone[0] *= -1
            measurements.append([i, cone[0], cone[1]])

    odom = np.array(odom)
    # print(odom)
    odom[:, [0, 1, 2]] = odom[:, [1, 0, 2]]
    odom[:, 1] +=2
    odom[:, 0] *= -1
    odom[:, 2] += np.pi/2
    odom[:, 2] = pi_2_pi(odom[:, 2])

    return odom, np.array(measurements)


track = np.load("../track.npy")


with open("odom_measurements_compare.json") as f:
    data = json.load(f)

odom, measurements = process_compare(data)

distances = []

i = 0
for (i, x, y) in measurements:
    print(i)
    min_dist = math.inf
    min_t = None

    for j, t in enumerate(track):
        dist = np.linalg.norm([x - t[0], y - t[1]])
        if dist < min_dist:
            min_dist = dist
            min_t = j

    distances.append([min_dist, min_t])


distances = np.array(distances)

print("old", len(measurements))

measurements = measurements[distances[:, 0] < 3]
distances = distances[distances[:, 0] < 3]

print("new", len(measurements))


data_known = []

for i in range(len(odom)):
    ms = measurements[measurements[:, 0] == i][:, [1, 2]]
    ds = distances[measurements[:, 0] == i][:, 1]
    ms = [[x, y, j] for ((x,y), j) in zip(ms, ds)]

    data_known.append({
        "true_odom": odom[i].tolist(),
        "measurements": ms
    })


with open("data_known.json", "w") as f:
    json.dump(data_known, f)

# lines = []
# N = len(measurements[:1000])

# for i in range(len(measurements[:1000])):
#     print(i, N)
#     _, x ,y = measurements[i]
#     _, tx, ty = distances[i]
#     plt.plot([x, tx], [y, ty], c="red", linestyle="dashed")


plt.scatter(measurements[:, 1], measurements[:, 2], c="orange", s=3)

plt.scatter(track[:, 0], track[:, 1], marker=(7, 1, 0), c="blue", s=35)
plt.axis("equal")

plt.show()