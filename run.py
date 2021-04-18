import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit


if __name__ == "__main__":
    from config_utias import config
    from fastslam_utias import run_SLAM
    # from config_jacobian_fsonline import config
    # from fastslam_jacobian_fsonline_real import run_SLAM
    # from config_jacobian_square import config
    # from fastslam_jacobian import run_SLAM

    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    for N, THREADS in zip([4, 8, 16, 32, 64, 128, 256, 512, 1024], [4, 8, 16, 32, 64, 128, 256, 512, 512]):
        config.N = N
        config.THREADS = THREADS

        seeds = np.arange(20)
        deviations = []
        for seed in seeds:
            deviation = run_SLAM(config, plot=False, seed=seed)
            deviations.append(deviation)

        print("Result: ", np.mean(deviations), np.median(deviations), np.std(deviations))

        output = {
            "result": [np.mean(deviations), np.median(deviations), np.std(deviations)],
        }

        fname = f"figs_utias/1_result_{config.DATASET}_{config.ROBOT}_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}.json"
        # fname = f"figs_jacobi_dist/2_result_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}.json"

        with open(fname, "w") as f:
            json.dump(output, f)

