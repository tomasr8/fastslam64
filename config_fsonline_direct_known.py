import json
import numpy as np
from numpy.testing._private.utils import measure
from utils import dotify
import math

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def process(data):
    odom = []
    measurements = []

    for i in range(len(data)):
        x, y, theta = data[i]["true_odom"]
        theta = pi_2_pi(theta)
        odom.append([x, y, theta])

        measurements.append(data[i]["measurements"])

    return np.array(odom), measurements


with open("fsonline/odom/data_known.json") as f:
    data = json.load(f)

true_odom, measurements = process(data)

xavier_odom = np.load("fsonline/odom/fixed_odom.npy")

config = {
    "SEED": 9,
    "N": 256, # number of particles
    "DT": 1.0,
    "THREADS": 256, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024, # available GPU heap size
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