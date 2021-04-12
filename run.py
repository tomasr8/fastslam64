import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit


if __name__ == "__main__":
    from config_jacobian_square import config
    from fastslam_jacobian import run_SLAM
    # from config_model_square import config
    # from fastslam_model import run_SLAM
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    seeds = np.arange(100)
    deviations = []
    for seed in seeds:
        deviation = run_SLAM(config, plot=False, seed=seed)
        deviations.append(deviation)

    print("Result: ", np.mean(deviations), np.median(deviations), np.std(deviations))

