import matplotlib.pyplot as plt
import numpy as np
import json

with open("true_odom.json") as f:
    true_odom = json.load(f)

with open("xavier_odom.json") as f:
    xavier_odom = json.load(f)


print(len(true_odom))
print(len(xavier_odom))


td = np.zeros((len(true_odom), 3), dtype=np.float64)
xd = np.zeros((len(xavier_odom), 3), dtype=np.float64)

for i in range(len(true_odom)):
    td[i, 0] = true_odom[i]['position'][0]
    td[i, 1] = true_odom[i]['position'][1]
    td[i, 2] = true_odom[i]['orientation']

for i in range(len(xavier_odom)):
    xd[i, 0] = xavier_odom[i]['position'][0]
    xd[i, 1] = xavier_odom[i]['position'][1]
    xd[i, 2] = xavier_odom[i]['orientation']


td[:, [0, 1, 2]] = td[:, [1, 0, 2]]
td[:, 1] +=2
td[:, 0] *= -1
td[:, 2] = np.arctan2(np.sin(td[:, 2]), np.cos(td[:, 2]))

xd[:, [0, 1, 2]] = xd[:, [1, 0, 2]]
xd[:, 1] +=2
xd[:, 0] *= -1
xd[:, 2] = np.arctan2(np.sin(xd[:, 2]), np.cos(xd[:, 2]))

# plt.plot(td[:, 0], td[:, 1])
# plt.plot(xd[:, 0], xd[:, 1])

plt.plot(td[:, 2], c="green")
plt.plot(xd[:, 2], c="orange")

print(np.mean(td[: 2] - xd[: 2]))


print(td[80:83])
print(xd[80:83])


# m = np.sort(np.abs(np.arctan2(np.sin(td[:,2]-xd[:,2]), np.cos(td[:,2]-xd[:,2]))))
# m = np.rad2deg(m)

# print(np.flip(m)[:10])

plt.show()