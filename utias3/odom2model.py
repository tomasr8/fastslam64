import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def motion_model(pose, u):
    [x, y, theta] = pose

    theta += u[0]
    theta = pi_2_pi(theta)

    x += np.cos(theta) * u[1]
    y += np.sin(theta) * u[1]

    theta += u[2]
    theta = pi_2_pi(theta)

    return [x, y, theta]


np.random.seed(0)

odom = np.load("npy/odom_1_50hz.npy")
N = len(odom)


control = np.zeros((len(odom), 4), dtype=np.float)
control[0] = [0, 0, 0, odom[0, -1]]

poses = [list(odom[0, 1:])]

for i in range(1, N):
    start = odom[i-1]
    end = odom[i]
    
    stamp = end[-1]

    angle_between_positions = pi_2_pi(np.arctan2(end[1] - start[1], end[0] - start[0]))
    dist = np.linalg.norm(start[:2] - end[:2])

    # print(f"angle between: {np.rad2deg(angle_between_positions)}")

    angle = pi_2_pi(end[2] - start[2])
    # print(f"dist: {dist}")
    target = pi_2_pi(angle_between_positions - start[2])
    # print(f"target: {np.rad2deg(target)}")
    # print(f"target2: {pi_2_pi(end[2] - angle_between_positions)}")

    control[i] = [
        stamp
        target,
        dist,
        pi_2_pi(end[2] - angle_between_positions),
    ]

    poses.append(motion_model(poses[-1], control[i]))

    # poses.append([
    #     poses[-1][0] + np.cos(poses[-1][2]) * dist,
    #     poses[-1][1] + np.sin(poses[-1][2]) * dist,
    #     pi_2_pi(poses[-1][2] + angle)
    # ])

    # print(poses)


poses = np.array(poses)
print(odom[:N].shape)
print(poses.shape)
# print("poses", poses)
# print("odom", odom[:N])
# plt.scatter(odom[:N, 0], odom[:N, 1], color="green", s=5)
# plt.scatter(poses[:, 0], poses[:, 1], color="orange", s=5)

# plt.show()

np.save("control.npy", control)