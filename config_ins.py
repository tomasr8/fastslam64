import json
import numpy as np
from lib.utils import dotify
import math

def get_heading(o):
    roll = np.arctan2(
        2.0*(o[0]*o[1] + o[3]*o[2]),
        o[3]*o[3] + o[0]*o[0] - o[1]*o[1] - o[2]*o[2]
    )
    return roll + np.pi/2

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def lidar_to_global(cones, x, y, heading):
    cones = 1 * cones.copy()
    cones[:, 1] *= -1

    R = np.float32([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])

    cones = np.matmul(R, cones.T).T
    cones[:, 0] += y
    cones[:, 1] += x
    cones[:, [0, 1]] = cones[:, [1, 0]]

    return cones

def process(data, N):
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
            cones = lidar_to_global(cones[:, :2], x, y, theta)
            measurements.append(cones)
        else:
            measurements.append([])

    odom = np.array(odom)

    return odom, measurements


with open("/home/tomas/Desktop/eforce/ins_funkcni/slam.json") as f:
    data = json.load(f)

data = data[750:]
N = len(data)
odom, measurements = process(data, N)

config = {
    "SEED": 9,
    "N": 512, # number of particles
    "DT": 1.0,
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 2 * 100000 * 1024, # available GPU heap size
    "THRESHOLD": 4.0,
    "sensor": {
        "RANGE": 30,
        "FOV": np.pi,
        "VARIANCE": [0.1 ** 2, np.deg2rad(1) ** 2],
        "MAX_MEASUREMENTS": 100, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements,
        "MISS_PROB": 0
    },
    "ODOMETRY": odom,
    "ODOMETRY_VARIANCE": [0.2 ** 2, 0.2 ** 2, np.deg2rad(1) ** 2],
    # "LANDMARKS": np.load("accel_landmarks.npy").astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 1000, # upper bound on the total number of landmarks in the environment
    "START_POSITION": odom[0].copy()
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS
