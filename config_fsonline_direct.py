import json
import numpy as np
from numpy.testing._private.utils import measure
from utils import dotify
import math

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def lidar_to_global(cones, x, y, heading):
    cones = np.array(cones).copy()
    cones[:, 0] += 1.2

    R = np.float32([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])


    cones = np.matmul(R, cones.T).T
    cones[:, 0] += x
    cones[:, 1] += y

    return cones


def process(data, N):
    odom = []
    measurements = []

    for i in range(N):
        x, y, theta = data[i]["true_odom"]
        theta = pi_2_pi(theta)
        # x += 0.4*np.cos(theta)
        # y += 0.4*np.sin(theta)
        odom.append([x, y, theta])

        cones = data[i]["measurements"]
        cones = lidar_to_global(cones, x, y, theta)

        measurements.append([])
        for cone in cones:
            cone[0], cone[1] = cone[1], cone[0]
            cone[1] += 2
            cone[0] *= -1
            measurements[-1].append(cone)


    # measurements = np.array(measurements)

    odom = np.array(odom)
    # print(odom)
    odom[:, [0, 1, 2]] = odom[:, [1, 0, 2]]
    odom[:, 1] +=2
    odom[:, 0] *= -1
    odom[:, 2] += np.pi/2
    odom[:, 2] = pi_2_pi(odom[:, 2])

    # measurements[:, [0, 1]] = measurements[:, [1, 0]]
    # measurements[:, 1] +=2
    # measurements[:, 0] *= -1

    return odom, measurements


with open("fsonline/odom/odom_measurements_compare.json") as f:
    data = json.load(f)

N = len(data)
true_odom, measurements = process(data, N)

xavier_odom = np.load("fsonline/odom/fixed_odom.npy")

config = {
    "SEED": 9,
    "N": 1024, # number of particles
    "DT": 1.0,
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 2 * 100000 * 1024, # available GPU heap size
    "THRESHOLD": 3.25,
    "sensor": {
        "RANGE": 30,
        "FOV": np.pi,
        "VARIANCE": [0.2 ** 2, np.deg2rad(15.0) ** 2],
        "MAX_MEASUREMENTS": 100, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements,
        "MISS_PROB": 0
    },
    "ODOMETRY": true_odom,
    "EST_ODOMETRY": xavier_odom,
    "ODOMETRY_VARIANCE": [0.1, 0.1, 0.01],
    "LANDMARKS": np.load("fsonline/track.npy").astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 1000, # upper bound on the total number of landmarks in the environment
    "START_POSITION": true_odom[0].copy()
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS