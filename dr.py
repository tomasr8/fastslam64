import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

np.random.seed(0)
# OFFSET_YAW_RATE_NOISE = 0.01

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = F @ x + B @ u

    x[2, 0] = pi_2_pi(x[2, 0])

    return x


DT = 1.0
# R_sim = np.diag([0.05, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([0.05, 0.01]) ** 2


xt = np.array([1.0, 1.0, 0]).reshape(3, 1)
xd = np.array([1.0, 1.0, 0]).reshape(3, 1)
hxDR = [xd]
hxTrue = [xt]
fig, ax = plt.subplots()

control = np.load("square/control.npy")
control[:, [0, 1]] = control[:, [1, 0]]

print(control[:10])

for i in range(1500):
    # v = 1.0/10  # [m/s]
    # yaw_rate = 0.1/10  # [rad/s]
    # u = np.array([v, yaw_rate]).reshape(2, 1)
    u = control[i].reshape(2, 1)

    hxTrue.append(motion_model(hxTrue[-1], u))

    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5# + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)

    hxDR.append(motion_model(hxDR[-1], ud))

hxDR = np.array(hxDR)
hxTrue = np.array(hxTrue)


# dead_reckoning = [[1.0, 1.0, 0]]
# for i in range(1500):
#     ua = control[i, 1]# + np.random.randn() * (R_sim[1,1] ** 0.5)
#     ub = control[i, 0]# + np.random.randn() * (R_sim[0,0] ** 0.5)

#     angle = pi_2_pi(dead_reckoning[-1][2] + ua)

#     x = dead_reckoning[-1][0] + np.cos(dead_reckoning[-1][2]) * ub
#     y = dead_reckoning[-1][1] + np.sin(dead_reckoning[-1][2]) * ub

#     dead_reckoning.append([ x, y, angle ])


dead_reckoning = [[1.0, 1.0, 0]]
for i in range(1500):
    ua = control[i, 1] + np.random.randn() * (R_sim[1,1] ** 0.5)
    ub = control[i, 0] + np.random.randn() * (R_sim[0,0] ** 0.5)

    dead_reckoning.append([
        dead_reckoning[-1][0] + (ub) * np.cos(dead_reckoning[-1][2]),
        dead_reckoning[-1][1] + (ub) * np.sin(dead_reckoning[-1][2]),
        pi_2_pi(dead_reckoning[-1][2] + (ua))
    ])

dead_reckoning = np.array(dead_reckoning)


plt.plot(dead_reckoning[:, 0], dead_reckoning[:, 1], "-r")
plt.plot(hxTrue[:, 0], hxTrue[:, 1], "-b")
plt.plot(hxDR[:, 0], hxDR[:, 1], "-k")

plt.axis("equal")

plt.show()