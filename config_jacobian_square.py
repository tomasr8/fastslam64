import numpy as np
from utils import dotify

config = {
    "SEED": 7,
    "N": 64,  # number of particles
    "DT": 1.0,
    "THREADS": 64,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 1.6,
    "sensor": {
        "RANGE": 8,
        "FOV": 0.75*np.pi,
        "MISS_PROB": 0,  # probability landmark in range will be missed
        "VARIANCE": [0.15 ** 2, np.deg2rad(1.0) ** 2],
        "MAX_MEASUREMENTS": 50  # upper bound on the total number of simultaneous measurements
    },
    # "gps": {
    #     "VARIANCE": [0.1 ** 2, 0.1 ** 2, np.deg2rad(1.0) ** 2],
    #     "RATE": 50
    # },
    "GROUND_TRUTH": np.load("square/odom.npy"),
    "CONTROL": np.load("square/control.npy"),
    "CONTROL_VARIANCE": [np.deg2rad(5.0) ** 2, 0.15 ** 2],
    # "CONTROL_VARIANCE": [np.deg2rad(3.0) ** 2, 0.1 ** 2],
    # "CONTROL_VARIANCE": [np.deg2rad(0.5) ** 2, 0.05 ** 2],
    "DEAD_RECKONING": np.load("square/dead_reckoning.npy"),
    "LANDMARKS": np.loadtxt("landmarks_square.txt").astype(np.float64),  # landmark positions
    "MAX_LANDMARKS": 500,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": np.array([1, 1, 0], dtype=np.float64)
}


config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS