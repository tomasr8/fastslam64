import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

np.random.seed(0)

N = 360
M = 50

odom = np.vstack((
    np.array([
        np.linspace(1, 19, num=N),
        np.ones(shape=N),
        np.zeros(shape=N)
    ]).T,
    np.array([
        19 * np.ones(shape=M),
        np.ones(shape=M),
        np.linspace(0, np.pi/2, num=M)
    ]).T,

    np.array([
        19 * np.ones(shape=N),
        np.linspace(1, 19, num=N),
        np.pi/2 * np.ones(shape=N)
    ]).T,
    np.array([
        19 * np.ones(shape=M),
        19 * np.ones(shape=M),
        np.linspace(np.pi/2, np.pi, num=M)
    ]).T,

    np.array([
        np.flip(np.linspace(1, 19, num=N)),
        19 * np.ones(shape=N),
        np.pi * np.ones(shape=N)
    ]).T,
    np.array([
        np.ones(shape=M),
        19 * np.ones(shape=M),
        np.linspace(np.pi, 1.5*np.pi, num=M)
    ]).T,

    np.array([
        np.ones(shape=N),
        np.flip(np.linspace(1, 19, num=N)),
        1.5*np.pi * np.ones(shape=N)
    ]).T,
    np.array([
        np.ones(shape=M),
        np.ones(shape=M),
        np.linspace(1.5*np.pi, 2*np.pi, num=M)
    ]).T
))

for i in range(len(odom)):
    odom[i, 2] = wrap_angle(odom[i, 2])

odom = np.vstack((
    odom.copy(),
    odom.copy()
))


control = np.zeros((len(odom), 2), dtype=np.float)
control[0] = [0, 0]

Q = [np.deg2rad(0.5) ** 2, 0.05 ** 2]

for i in range(1, len(odom)):
    start = odom[i-1]
    end = odom[i]
    
    angle = wrap_angle(end[2] - start[2])
    # angle += np.random.randn() * (Q[0] ** 0.5)
    angle = wrap_angle(angle)

    dist = np.linalg.norm(start[:2] - end[:2])
    
    # if dist > 0.1:
        # dist += np.random.randn() * (Q[1] ** 0.5)

    control[i] = [angle, dist]


dead_reckoning = [odom[0]]
for i in range(1, len(odom)):
    ua = control[i, 0]# + np.random.randn() * (Q[0] ** 0.5)
    ub = control[i, 1]# + np.random.randn() * (Q[1] ** 0.5)

    angle = wrap_angle(dead_reckoning[-1][2] + ua)

    x = dead_reckoning[-1][0] + np.cos(dead_reckoning[-1][2]) * ub
    y = dead_reckoning[-1][1] + np.sin(dead_reckoning[-1][2]) * ub

    dead_reckoning.append([ x, y, angle ])


dead_reckoning = np.array(dead_reckoning)


fig, ax = plt.subplots()

ax.scatter(odom[:, 0], odom[:, 1], s=2)
ax.scatter(dead_reckoning[:, 0], dead_reckoning[:, 1], s=2)

# np.save("odom.npy", odom)
# np.save("control.npy", control)
# np.save("dead_reckoning.npy", dead_reckoning)


plt.show()