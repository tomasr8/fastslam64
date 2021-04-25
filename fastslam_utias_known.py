import math
import time
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from particle3 import FlatParticle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit
from cuda.update_known import load_cuda_modules
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def to_coords(range, bearing, theta):
    return [range * np.cos(bearing + theta), range * np.sin(bearing + theta)]

def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        seed = config.SEED

    print(f"Setting seed: {seed}")    
    np.random.seed(seed)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        fig.tight_layout()

    cuda_modules = config.modules

    memory = CUDAMemory(config)
    weights = np.zeros(config.N, dtype=np.float64)

    cuda_modules = load_cuda_modules(
        THREADS=config.THREADS,
        PARTICLE_SIZE=config.PARTICLE_SIZE,
        N_PARTICLES=config.N
    )

    cuda.memcpy_htod(memory.cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(memory.particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(seed), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")

    # dead_reckoning = [config.START_POSITION]

    for i, (g, o) in enumerate(zip(config.GROUND_TRUTH, config.CONTROL)):
        stats.start_measuring("Loop")
        print(f"{i}/{config.CONTROL.shape[0]}")
        # print(o)

        # o[2] = 0.7*o[2]
        # print(i, g)
        # if i % 100 == 0:
            # print(i)

        # if i > 1000:
        #     break

        stats.start_measuring("Measurement")

        # ua = o[2]
        # ub = o[1]
        # dead_reckoning.append([
        #     dead_reckoning[-1][0] + config.DT * ub * np.cos(dead_reckoning[-1][2]),
        #     dead_reckoning[-1][1] + config.DT * ub * np.sin(dead_reckoning[-1][2]),
        #     wrap_angle(dead_reckoning[-1][2] + config.DT * ua)
        # ])

        t = g[0]
        measurements = config.sensor.MEASUREMENTS[config.sensor.MEASUREMENTS[:, 0] == t]
        # print(measurements)
        # print("================")
        measurements = measurements[:, [2,3,1]].astype(np.float64)

        # if measurements.size != 0:
        #     print("====================================================")
        #     print("====================================================")
        #     print("====================================================")


        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(memory.measurements, measurements.copy())

        # ===== RESET WEIGHTS =======
        cuda_modules["resample"].get_function("reset_weights")(
            memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        if "gps" in config and i % config.gps.RATE == 0:
            cuda_modules["predict"].get_function("predict_from_imu")(
                memory.particles,
                np.float64(g[1]), np.float64(g[2]), np.float64(g[3]),
                np.float64(config.gps.VARIANCE[0] ** 0.5), np.float64(config.gps.VARIANCE[1] ** 0.5), np.float64(config.gps.VARIANCE[2] ** 0.5),
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )
        else:
            cuda_modules["predict"].get_function("predict_from_model")(
                memory.particles,
                np.float64(o[2]), np.float64(o[1]),
                np.float64(config.CONTROL_VARIANCE[0] ** 0.5), np.float64(config.CONTROL_VARIANCE[1] ** 0.5),
                np.float64(config.DT),
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )


        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(config.N//config.THREADS),
            memory.measurements,
            np.int32(config.N), np.int32(len(measurements)),
            memory.cov,
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )

        rescale(cuda_modules, config, memory)
        estimate =  get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose(g[1:].tolist(), estimate.tolist())
        
        if plot and measurements.size > 0:# and i % 10 == 0:
            cuda.memcpy_dtoh(particles, memory.particles)

            ax[0].clear()
            ax[1].clear()

            # plot_sensor_fov(ax[0], g[1:], config.sensor.RANGE, config.sensor.FOV)
            # plot_sensor_fov(ax[1], g[1:], config.sensor.RANGE, config.sensor.FOV)

            ax[0].text(0, 5.5, f"Iteration: {i}")

            measurements = [to_coords(r, b, g[3]) for r, b, _ in measurements]
            measurements = np.array(measurements)

            # if measurements.size > 0:
            #     print(measurements + g[1:3])
            #     print()
            # print(estimate.tolist())


            if(measurements.size != 0):
                plot_connections(ax[0], g[1:], measurements + g[1:3])

            plot_landmarks(ax[0], config.LANDMARKS[:, 1:], color="blue", zorder=100)
            # plot_history(ax[0], np.array(dead_reckoning)[::50], color='purple')
            plot_history(ax[0], stats.ground_truth_path[::50], color='green')
            plot_history(ax[0], stats.predicted_path[::50], color='orange')
            plot_particles_weight(ax[0], particles)

            if(measurements.size != 0):
                plot_measurement(ax[0], g[1:3], measurements, color="orange", zorder=103)

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], config.LANDMARKS[:, 1:], color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)


        if i == config.CONTROL.shape[0]-1:
            cuda.memcpy_dtoh(particles, memory.particles)
            best = np.argmax(FlatParticle.w(particles))
            best_covariances = FlatParticle.get_covariances(particles, best)
            best_landmarks = FlatParticle.get_landmarks(particles, best)


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        stats.stop_measuring("Loop")

    stats.summary()


    if not plot:
        output = {
            "ground": stats.ground_truth_path,
            "predicted": stats.predicted_path,
            # "dead_reckoning": [list(pos) for pos in dead_reckoning],
            "landmarks": config.LANDMARKS[:, 1:].tolist(),
            "map": [list(lm) for lm in best_landmarks],
            "map_covariance": [cov.tolist() for cov in best_covariances]
        }

        fname = f"figs_utias/2_known_data_{config.DATASET}_{config.ROBOT}_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}_{seed}.json"

        with open(fname, "w") as f:
            json.dump(output, f)

        fig, ax = plt.subplots()
        # plot_history(ax, dead_reckoning[::100], color='purple', linewidth=0.3, markersize=0.5)
        plot_history(ax, stats.ground_truth_path[::100], color='green', linewidth=0.3, markersize=0.5)
        plot_history(ax, stats.predicted_path[::100], color='orange', linewidth=0.3, markersize=0.5)
        plot_landmarks(ax, config.LANDMARKS[:, 1:], color="blue")
        plot_map(ax, best_landmarks, color="orange", marker="o")
        for i, landmark in enumerate(best_landmarks):
            plot_confidence_ellipse(ax, landmark, best_covariances[i], n_std=3)

        fname = f"figs_utias/2_known_plot_{config.DATASET}_{config.ROBOT}_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}_{seed}.png"
        plt.savefig(fname)

    memory.free()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_utias import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    run_SLAM(config, plot=True)