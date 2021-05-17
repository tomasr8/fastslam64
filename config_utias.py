import numpy as np
from lib.utils import dotify

T = 1000

ground = np.load(f"utias/npy/ground.npy")
odom = np.load(f"utias/npy/odom.npy")
measurements = np.load(f"utias/npy/measurements.npy")
landmarks = np.load(f"utias/landmarks.npy")

ground = ground[ground[:, 0] < T]
odom = odom[odom[:, 0] < T]
measurements = measurements[measurements[:, 0] < T]

config = {
    "SEED": 2,
    "N": 512,  # number of particles
    "DT": 0.015,
    "THREADS": 512,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    "THRESHOLD": 1.3,
    "sensor": {
        "RANGE": 6,
        "FOV": 2*np.pi,
        "VARIANCE": [8*(0.15 ** 2), 2*(np.deg2rad(2.0) ** 2)],
        "MAX_MEASUREMENTS": 20, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements.astype(np.float64),
    },
    "CONTROL": odom.astype(np.float64),
    "CONTROL_VARIANCE": [np.deg2rad(8.0) ** 2, 0.1 ** 2],
    "GROUND_TRUTH": ground.astype(np.float64),
    "LANDMARKS": landmarks.astype(np.float64),  # landmark positions
    "MAX_LANDMARKS": 100,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": ground[0, 1:].astype(np.float64)
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS
