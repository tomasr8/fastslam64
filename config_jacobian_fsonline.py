import json
import numpy as np
from utils import dotify
from cuda.update_jacobian_dist import load_cuda_modules


odometry = np.load("fsonline/full_pipeline/odom.npy").astype(np.float64)
odometry[:, 2] += np.pi/2
control = np.load("fsonline/full_pipeline/control.npy").astype(np.float64)

with open("fsonline/full_pipeline/detections_converted.json") as f:
    measurements = json.load(f)

N = 0

config = {
    "SEED": 8,
    "N": 4096, # number of particles
    "DT": 1.0,
    "THREADS": 512, # number threads in a block
    "GPU_HEAP_SIZE_BYTES": 100000 * 1024, # available GPU heap size
    "THRESHOLD": 3.0,
    "sensor": {
        "RANGE": 30,
        "FOV": np.pi,
        "VARIANCE": [0.15 ** 2, np.deg2rad(1.0) ** 2],
        "MAX_MEASUREMENTS": 100, # upper bound on the total number of simultaneous measurements
        "MEASUREMENTS": measurements,
        "MISS_PROB": 0
    },
    # "gps": {
    #     "VARIANCE": [0.1 ** 2, 0.1 ** 2, np.deg2rad(1.0) ** 2],
    #     "RATE": 10
    # },
    "ODOMETRY": odometry[N:],
    "ODOMETRY_VARIANCE": [0.1, 0.1, 0.001],
    "CONTROL": control[N:],
    "CONTROL_VARIANCE": [np.deg2rad(0.25) ** 2, 0.05 ** 2],
    "LANDMARKS": np.load("fsonline/track.npy").astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 500, # upper bound on the total number of landmarks in the environment
    "START_POSITION": odometry[N, :3]
}

config = dotify(config)

config.sensor.COVARIANCE = \
    np.diag(config.sensor.VARIANCE).astype(np.float64)

config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS

config.modules = load_cuda_modules(
    THREADS=config.THREADS,
    PARTICLE_SIZE=config.PARTICLE_SIZE,
    N_PARTICLES=config.N
)