import numpy as np
from lib.utils import dotify

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
        "MISS_PROB": 0
    },
    "ODOMETRY_VARIANCE": [0.2 ** 2, 0.2 ** 2, np.deg2rad(1) ** 2],
    # "LANDMARKS": np.load("accel_landmarks.npy").astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 1000, # upper bound on the total number of landmarks in the environment
}


config = dotify(config)
config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 8*config.MAX_LANDMARKS