import matplotlib.pyplot as plt
import numpy as np
import json

with open("odom.json") as f:
    odom = json.load(f)

print(len(odom))

odometry = np.zeros((len(odom), 4), dtype=np.float32)
TIME_START = 1614630690


for i, o in enumerate(odom):
    odometry[i, 0] = o['position'][0]
    odometry[i, 1] = o['position'][1]

    [x, y, z, w] = o['orientation']

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    odometry[i, 2] = yaw

    stamp = o['stamp']


    # print(stamp['secs'])

    odometry[i, 3] = (stamp['secs'] - TIME_START) * 1000 + np.round(stamp['nsecs'] / 1e6)
    # odometry[i, 4] = stamp['nsecs']

    # if i > 3:
    #     break


odometry[:, [0, 1, 2]] = odometry[:, [1, 0, 2]]
odometry[:, 1] +=2
odometry[:, 0] *= -1

# TIME_START = 1614630690.0

# odometry[:, 4] /= 1e6
# odometry[:, 4] = np.round(odometry[:, 4])

# a = int(odometry[0, 3])
# b = int(TIME_START)
# print(a, b, a-b)

# print(odometry[0, 3], TIME_START, odometry[0, 3] - TIME_START)
# print(1614630700.0-1614630690)


# odometry[:, 3] -= TIME_START

# print(odometry[:, 3])

print(np.min(odometry[:, 3]), np.max(odometry[:, 3]))

print(odometry[:, 3])
print(odometry.shape)

np.save("odom.npy", odometry)