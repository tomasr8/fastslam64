import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

np.random.seed(0)

V = 0.05
W = np.deg2rad(5)

waypoints = [
    [2, 6],
    [4.5, 1.0],
    [8, 1.8],
    [7, 6],
    [4.2, 8.2],
    [1, 6.3],
    [2.2, 1.8],
    [5, 4.3],
    [8.9, 2.3],
    [8.3, 7.7],
    [5.0, 8.9],
    [3.4, 5.8],
    [2, 6]
]
odom = [[waypoints[0][0], waypoints[0][1], 0]]
indices = [0]

for i in range(len(waypoints) -1):
    start = waypoints[i]
    end = waypoints[i+1]
    # print(start, end)
    # print(i)

    angle = wrap_angle(np.arctan2(end[1] - start[1], end[0] - start[0]))

    # print(np.rad2deg(angle), np.rad2deg(odom[-1][2]), np.rad2deg(wrap_angle(angle - wrap_angle(odom[-1][2]))))

    n = int(np.abs(wrap_angle(angle - wrap_angle(odom[-1][2]))) / W)

    for a in np.linspace(odom[-1][2], angle, num=n):
        odom.append([odom[-1][0], odom[-1][1], wrap_angle(a)])

    n = int(np.linalg.norm([end[1] - start[1], end[0] - start[0]]) / V)

    for x, y, in zip(np.linspace(start[0], end[0], num=n), np.linspace(start[1], end[1], num=n)):
        odom.append([x, y, angle])

    indices.append(len(odom) - 1)

print(indices)

odom = np.array(odom)

for i in range(len(odom)):
    odom[i, 2] = wrap_angle(odom[i, 2])


control = np.zeros((len(odom), 2), dtype=np.float)
control[0] = [0, 0]

for i in range(1, len(odom)):
    start = odom[i-1]
    end = odom[i]
    
    angle = wrap_angle(end[2] - start[2])
    angle = wrap_angle(angle)

    dist = np.linalg.norm(start[:2] - end[:2])

    control[i] = [angle, dist]



landmarks = np.load("landmarks.npy")

fig, ax = plt.subplots()
ax.scatter(landmarks[:, 0], landmarks[:, 1], s=7)
ax.scatter(odom[:, 0], odom[:, 1], s=2)

# np.save("odom.npy", odom)
# np.save("control.npy", control)
# np.save("dead_reckoning.npy", dead_reckoning)


plt.show()