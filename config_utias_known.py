import numpy as np
from utils import dotify

START = 0
T = 1000.0
robot = 1
dataset = 3

# ground = np.load(f"utias/npy/ground_{robot}_50hz.npy")[START:]
# odom = np.load(f"utias/npy/odom_{robot}_50hz.npy")[START:]
# measurements = np.load(f"utias/npy_fixed/measurements_rb_{robot}_50hz.npy")
# landmarks = np.load("utias/landmarks.npy")

ground = np.load(f"utias{dataset}/npy/ground_{robot}_50hz.npy")[START:]
odom = np.load(f"utias{dataset}/npy/odom_{robot}_50hz.npy")[START:]
measurements = np.load(f"utias{dataset}/npy/measurements_{robot}_50hz.npy")
landmarks = np.load(f"utias{dataset}/landmarks.npy")

ground = ground[ground[:, 0] < T]
odom = odom[odom[:, 0] < T]
measurements = measurements[measurements[:, 0] < T]

config = {
    "ROBOT": robot,
    "DATASET": dataset,
    "SEED": 2,
    "N": 128,  # number of particles
    "DT": 0.015,
    "THREADS": 128,  # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024,  # available GPU heap size
    # "THRESHOLD": 1.2,
    "THRESHOLD": 1.0,
    "sensor": {
        "RANGE": 6,
        "FOV": 2*np.pi,
        # "VARIANCE": [0.12 ** 2, np.deg2rad(0.8) ** 2], # Robot 2?
        # "VARIANCE": [0.15 ** 2, np.deg2rad(1.0) ** 2], # Robot 1
        "VARIANCE": [0.6 ** 2, np.deg2rad(5.0) ** 2],
        "MAX_MEASUREMENTS": 50, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements.astype(np.float64),
    },
    # "gps": {
    #     "VARIANCE": [0.1 ** 2, 0.1 ** 2, np.deg2rad(1.0) ** 2],
    #     "RATE": 50
    # },
    "CONTROL": odom.astype(np.float64),
    # "CONTROL_VARIANCE": [np.deg2rad(20.0) ** 2, 0.2 ** 2], #robot 2
    # "CONTROL_VARIANCE": [np.deg2rad(10.0) ** 2, 0.1 ** 2], # robot 1
    "CONTROL_VARIANCE": [np.deg2rad(15.0) ** 2, 0.2 ** 2], # robot 1

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
