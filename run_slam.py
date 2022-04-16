import json
import numpy as np
import math
from slam import Slam


def get_heading(o):
    # The heading given by the INS is encoded in the roll component
    # It also needs to be negated for some reason,
    # otherwise it turns in the wrong direction.
    roll = np.arctan2(
        2.0 * (o[0] * o[1] + o[3] * o[2]),
        o[3] * o[3] + o[0] * o[0] - o[1] * o[1] - o[2] * o[2]
    )
    return -roll  # - np.deg2rad(5)


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def local_to_global(cones, x, y, heading):
    """Convert cones from the car frame to the global frame"""

    cones = cones.copy()

    R = np.float32([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])

    cones = np.matmul(R, cones.T).T
    cones[:, 0] += x
    cones[:, 1] += y

    return cones


class Subset:
    def __init__(self, parent, rank):
        self.parent = parent
        self.rank = rank


def find(subsets, node):
    if subsets[node].parent != node:
        subsets[node].parent = find(subsets, subsets[node].parent)
    return subsets[node].parent


def union(subsets, u, v):
    if subsets[u].rank > subsets[v].rank:
        subsets[v].parent = u
    elif subsets[v].rank > subsets[u].rank:
        subsets[u].parent = v

    else:
        subsets[v].parent = u
        subsets[u].rank += 1


def _remove_duplicate_measurements(measurements, thresh):
    n = len(measurements)

    if n < 2:
        return measurements

    subsets = [Subset(p, 0) for p in range(n)]

    for i in range(n - 1):
        for j in range(i + 1, n):
            a = measurements[i][:2]
            b = measurements[j][:2]
            if np.linalg.norm(a - b) < thresh:
                union(subsets, i, j)

    distinct = set(find(subsets, i) for i in range(n))

    merged = []
    for i, p in enumerate(distinct):
        duplicates = [measurements[j] for j in range(n) if find(subsets, j) == p]
        merged.append(np.mean(duplicates, axis=0))

    return np.array(merged)


def remove_duplicate_measurements(measurements, thresh):
    """
    Removes simultaneous duplicate measurements with 'thresh' distance and
    replaces them with the mean.
    """

    old = measurements
    new = _remove_duplicate_measurements(measurements, thresh=thresh)
    while old.shape[0] != new.shape[0]:
        old = new
        new = _remove_duplicate_measurements(measurements, thresh=thresh)

    return new


def process(data, N):
    """Process a SLAM dataset into odometry (pose) and measurements"""
    odom = []
    measurements = []

    for i in range(N):
        x, y, _ = data[i]['position']
        o = data[i]['orientation']

        theta = get_heading(o)
        theta = pi_2_pi(theta)

        odom.append(((x,y),(o[0],o[1],o[2],o[3])))

        cones = np.array(data[i]["detections"])
        if cones.size > 0:
            colors = cones[:, 3]
            colors = np.reshape(colors, (colors.shape[0], 1))
            cones = local_to_global(cones[:, :2], x, y, theta)
            cones = np.hstack((cones, colors))
            cones = remove_duplicate_measurements(cones, thresh=1.5)
            # Each measurement has x & y global position and color        
            measurements.append(cones)
        else:
            measurements.append([])

    odom = np.array(odom)

    return odom, measurements


with open('td_02.json') as f:
    data = json.load(f)

data = data[80:]
N = len(data)
odom, measurements = process(data, N)

slam = Slam(start_position=odom[0].copy())

for i in range(odom.shape[0]):

    slam.set_odometry(odom[i])
    print(slam.set_measurements(measurements[i])[1])

