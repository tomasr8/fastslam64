import matplotlib.pyplot as plt
import numpy as np
import json

def min_dist(arr, stamp):
    arr = np.abs(arr - stamp)

    argmin = np.argmin(arr)
    print("min:", np.min(arr))
    if arr[argmin] < 100:
        return argmin
    else:
        return None


track = np.load("track.npy")

with open("full_pipeline/detections_converted.json") as f:
    detections = json.load(f)

odom = np.load("full_pipeline/odom.npy")


fig, ax = plt.subplots()
ax.scatter(track[:, 0], track[:, 1])

# for d in detections:
#     stamp = d['stamp']

#     o = odom[min_dist(odom[:, -1], stamp)]
#     print(stamp, o[-1])

#     points = np.array(d['points'])[:, :2]

#     points[:, [0, 1]] = points[:, [1, 0]]
#     points[:, 1] +=2
#     points[:, 0] *= -1

#     ax.scatter(points[:, 0], points[:, 1], c="g")
#     ax.scatter([o[0]], [o[1]], c="r")
#     plt.pause(0.0001)


stamps = np.array([d['stamp'] for d in detections])

print(np.min(stamps), np.max(stamps))
print(np.min(odom[:, -1]), np.max(odom[:, -1]))


for o in odom:
    stamp = o[-1]
    print(stamp)

    argmin = min_dist(stamps, stamp)
    if argmin is not None:
        print("plotting")
        d = detections[argmin]
        # print(stamp, d['stamp'])

        points = np.array(d['points'])[:, :2]

        points[:, [0, 1]] = points[:, [1, 0]]
        points[:, 1] +=2
        points[:, 0] *= -1

        ax.scatter(points[:, 0], points[:, 1], c="g")

    ax.scatter([o[0]], [o[1]], c="r")
    plt.pause(.01)
