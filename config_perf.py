import numpy as np
from utils import dotify

config = {
    "SEED": 12,
    "N": 512,  # number of particles
    "DT": 1.0,
    "THREADS": 512,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 0.008,
    "sensor": {
        "RANGE": 0.5,
        "FOV": 2*np.pi,
        "MISS_PROB": 0,  # probability landmark in range will be missed
        "VARIANCE": [0.05 ** 2, np.deg2rad(0.01) ** 2],
        "MAX_MEASUREMENTS": 1000  # upper bound on the total number of simultaneous measurements
    },
    "CONTROL": np.load("perf/control.npy"),
    "CONTROL_VARIANCE": [np.deg2rad(0.5) ** 2, 0.05 ** 2],
    # "CONTROL_VARIANCE": [np.deg2rad(3.0) ** 2, 0.1 ** 2],
    # "CONTROL_VARIANCE": [np.deg2rad(0.5) ** 2, 0.05 ** 2],
    # "LANDMARKS": np.load("circle/landmarks.npy").astype(np.float64),  # landmark positions
    "LANDMARKS": np.load("perf/landmarks_1000.npy").astype(np.float64),  # landmark positions

    "MAX_LANDMARKS": 5000,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": np.array([0, 0, 0], dtype=np.float64)
}


config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS