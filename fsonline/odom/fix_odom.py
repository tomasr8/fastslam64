import math
import numpy as np
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



with open("odom_measurements_compare.json") as f:
    data = json.load(f)

# A = 80
# B = 90
A = 0
# B = 156
B = 891
true_odom, x_odom, true_measurements, x_measurements = process_compare(data, A, B)

dist = []
for i in range(len(x_odom) - 1):
    dist.append(np.linalg.norm([
        x_odom[i, 0] - x_odom[i+1, 0],
        x_odom[i, 1] - x_odom[i+1, 1],
    ]))

# print(np.median(dist))
# print(np.mean(dist))

m_dist = 0.496


x_odom_fixed = x_odom.copy()
skip_next = False
for i in range(90, len(x_odom_fixed) -1):
    if skip_next:
        skip_next = False
        continue

    x, y, theta = x_odom_fixed[i]

    xp = x + m_dist * np.cos(theta)
    yp = y + m_dist * np.sin(theta)

    print("curr", [x, y, theta])
    print("xpyp", [xp, yp])

    x, y, _ = x_odom_fixed[i+1]
    print([x, y])


    d = np.linalg.norm([x - xp, y - yp])
    angle = np.arctan2(y - x_odom_fixed[i][1], x - x_odom_fixed[i][0])
    print("da", d, np.rad2deg(angle))
    if d > 0.2 and np.abs(angle) > np.pi/6:
        x_odom_fixed[i+1] = [xp, yp, x_odom_fixed[i+1][2]]
        skip_next = True
    # print(i, d)


np.save("fixed_odom.npy", x_odom_fixed)

for (x, y, theta) in x_odom:
    plt.plot([x, x+0.2*np.cos(theta)], [y, y+0.2*np.sin(theta)], c="orange")
    # plt.plot([x, x+15*np.cos(theta)], [y, y+15*np.sin(theta)], c="orange", linewidth=0.1)

for (x, y, theta) in x_odom_fixed:
    plt.plot([x, x+0.2*np.cos(theta)], [y, y+0.2*np.sin(theta)], c="purple")

plt.scatter(x_odom[:, 0], x_odom[:, 1], c="orange", s=7)
plt.scatter(x_odom_fixed[:, 0], x_odom_fixed[:, 1], c="purple", s=3)



# # true_measurements = true_measurements[true_measurements[:, 1] < 18]
# # x_measurements = x_measurements[x_measurements[:, 1] < 18]


# plt.scatter(true_measurements[:, 0], true_measurements[:, 1], c="purple", s=3)
# plt.scatter(x_measurements[:, 0], x_measurements[:, 1], c="orange", s=3)

# track = np.load("../track.npy")
# # track = track[(track[:, 0] > -5) & (track[:, 0] < 5)]
# # track = track[(track[:, 1] > 0) & (track[:, 1] < 11)]
# plt.scatter(track[:, 0], track[:, 1], c="blue", s=4)
plt.axis("equal")

plt.show()