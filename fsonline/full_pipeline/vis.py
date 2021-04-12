import matplotlib.pyplot as plt
import numpy as np
import json
import math

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

track = np.load("../track.npy")

odometry = np.load("odom.npy")
# odometry = odometry[4000:]
# odometry[:, [0, 1, 2]] = odometry[:, [1, 0, 2]]
# odometry[:, 1] +=2
# odometry[:, 0] *= -1

print(odometry[:10])

odometry[:, 2] += np.pi/2
odometry[:, 2] = pi_2_pi(odometry[:, 2])

print(odometry[:10])



control = np.load("control.npy")
print(odometry.shape)
print(control.shape)

# plt.scatter(track[:, 0], track[:, 1], s=3)
N = 10
plt.scatter(odometry[:N, 0], odometry[:N, 1], s=1, color="green")
plt.axis("equal")

hist = [odometry[0, :3]]
for i in range(N):
    u = control[i]

    angle = pi_2_pi(hist[-1][2] + u[0])
    hist.append([
        hist[-1][0] + np.cos(hist[-1][2]) * u[1],
        hist[-1][1] + np.sin(hist[-1][2]) * u[1],
        angle
    ])

hist = np.array(hist)
plt.scatter(hist[:, 0], hist[:, 1], color="purple", s=2)
print(control.shape)
print(odometry.shape)

plt.show()