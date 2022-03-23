import json
import numpy as np
from lib.utils import dotify
import math

def get_heading(o):
    # The heading given by the INS is encoded in the roll component
    # It also needs to be negated for some reason,
    # otherwise it turns in the wrong direction.
    roll = np.arctan2(
        2.0*(o[0]*o[1] + o[3]*o[2]),
        o[3]*o[3] + o[0]*o[0] - o[1]*o[1] - o[2]*o[2]
    )
    return -roll


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

    for i in range(n-1):
        for j in range(i+1, n):
            a = measurements[i][:2]
            b = measurements[j][:2]
            if np.linalg.norm(a-b) < thresh:
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

        odom.append([x, y, theta])

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


with open("aut_lidar.json") as f:
    data = json.load(f)

data = data[800:]
N = len(data)
odom, measurements = process(data, N)

config = {
    "SEED": 9,
    "N": 2048, # number of particles
    "DT": 1.0,
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 2 * 100000 * 1024, # available GPU heap size
    "THRESHOLD": 2.1,
    "sensor": {
        "RANGE": 13.0,
        "FOV": np.pi*0.85,
        # "VARIANCE": [0.25 ** 2, np.deg2rad(1) ** 2],
        "VARIANCE": [1.5 ** 2, np.deg2rad(5) ** 2],
        "MAX_MEASUREMENTS": 100, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements,
        "MISS_PROB": 0
    },
    "ODOMETRY": odom,
    # "ODOMETRY_VARIANCE": [0.1 ** 2, 0.1 ** 2, np.deg2rad(1) ** 2],
    "ODOMETRY_VARIANCE": [0.2 ** 2, 0.2 ** 2, np.deg2rad(1.5) ** 2],

    # "LANDMARKS": np.load("accel_landmarks.npy").astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 1000, # upper bound on the total number of landmarks in the environment
    "START_POSITION": odom[0].copy()
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 8*config.MAX_LANDMARKS
