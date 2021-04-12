import matplotlib.pyplot as plt
import numpy as np
import json

track = np.load("track.npy")

with open("full_pipeline/odom.json") as f:
    odom = json.load(f)

odometry = np.zeros((len(odom), 3), dtype=np.float32)

for i, o in enumerate(odom):
    odometry[i, 0] = o['position'][0]
    odometry[i, 1] = o['position'][1]

    [x, y, z, w] = o['orientation']

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    odometry[i, 2] = yaw


odometry[:, [0, 1, 2]] = odometry[:, [1, 0, 2]]
odometry[:, 1] +=2
odometry[:, 0] *= -1

np.save("full_pipeline/odom.npy", odometry)


#=================================
#=================================
#=================================

# odometry = np.load("odom.npy")

# blue = track[track[:, 2] == 0]
# yellow = track[track[:, 2] == 1]


# plt.scatter(blue[:, 0], blue[:, 1])
# plt.scatter(yellow[:, 0], yellow[:, 1])

# plt.scatter(odometry[::20, 0], odometry[::20, 1], s=1)
# plt.axis("equal")

# for i in range(0, len(odometry), 150):
#     angle = 90 + odometry[i, 2] * 180 / np.pi
#     plt.text(odometry[i, 0], odometry[i, 1], f"{angle:.2f}")


# plt.show()