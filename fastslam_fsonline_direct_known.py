import numpy as np
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
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate
from cuda.update_known import load_cuda_modules

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def rb2xy(pose, rb):
    [_, _, theta] = pose
    [r, b] = rb
    return [r * np.cos(b + theta), r * np.sin(b + theta)]

def xy2rb(pose, landmark):
    position = pose[:2]
    vector_to_landmark = np.array(landmark - position, dtype=np.float64)

    r = np.linalg.norm(vector_to_landmark)
    b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]
    b = wrap_angle(b)

    return r, b


def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        seed = config.SEED
    np.random.seed(seed)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].axis('scaled')
        ax[1].axis('scaled')

    cuda_modules = load_cuda_modules(
        THREADS=config.THREADS,
        PARTICLE_SIZE=config.PARTICLE_SIZE,
        N_PARTICLES=config.N
    )

    memory = CUDAMemory(config)
    weights = np.zeros(config.N, dtype=np.float64)

    cuda.memcpy_htod(memory.cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(memory.particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(seed), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")

    for i in range(config.ODOMETRY.shape[0]):
        stats.start_measuring("Loop")
        # print(f"iteration: {i}")

        stats.start_measuring("Measurement")

        pose = config.ODOMETRY[i].copy()
        pose[2] = wrap_angle(pose[2])
        est_pose = config.EST_ODOMETRY[i].copy()
        est_pose[2] = wrap_angle(est_pose[2])

        measurements = config.sensor.MEASUREMENTS[i]
        visible_measurements = []
        for m in measurements:
            x, y, j = m
            r, b = xy2rb(pose, [x, y])
            visible_measurements.append([r, b, j+1])
            
        visible_measurements = np.array(visible_measurements, dtype=np.float64)

        stats.stop_measuring("Measurement")

        # ===== RESET WEIGHTS =======
        cuda_modules["resample"].get_function("reset_weights")(
            memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda.memcpy_htod(memory.measurements, visible_measurements)

        cuda_modules["predict"].get_function("predict_from_imu")(
            memory.particles,
            np.float64(est_pose[0]), np.float64(est_pose[1]), np.float64(est_pose[2]),
            np.float64(config.ODOMETRY_VARIANCE[0] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[1] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[2] ** 0.5),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        # block_size = config.N if config.N < 32 else 32

        # print("N MEASUREMENTS", len(visible_measurements))

        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(config.N//config.THREADS),
            memory.measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            memory.cov, 
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )
        # cuda_modules["update"].get_function("update")(
        #     memory.particles, np.int32(1),
        #     memory.measurements,
        #     np.int32(config.N), np.int32(len(visible_measurements)),
        #     memory.cov, np.int32(config.MAX_LANDMARKS),
        #     block=(block_size, 1, 1), grid=(config.N//block_size, 1, 1)
        # )

        rescale(cuda_modules, config, memory)
        estimate = get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose(pose.tolist(), estimate.tolist())

        if plot:
            cuda.memcpy_dtoh(particles, memory.particles)

            ax[0].clear()
            ax[1].clear()
            ax[0].set_xlim([-160, 10])
            ax[0].set_ylim([-30, 50])
            ax[1].set_xlim([-160, 10])
            ax[1].set_ylim([-30, 50])
            ax[0].set_axis_off()
            ax[1].set_axis_off()

            plot_sensor_fov(ax[0], pose, config.sensor.RANGE, config.sensor.FOV)
            plot_sensor_fov(ax[1], pose, config.sensor.RANGE, config.sensor.FOV)

            visible_measurements = np.array([rb2xy(pose, m[:2]) for m in visible_measurements])

            if(visible_measurements.size != 0):
                plot_connections(ax[0], pose, visible_measurements + pose[:2])

            plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
            # plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], stats.ground_truth_path, color='green')
            plot_history(ax[0], stats.predicted_path, color='orange')
            plot_history(ax[0], config.ODOMETRY[:i], color='red')

            # plot_history(ax[0], dead_reckoning, color='purple')

            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], pose[:2], visible_measurements, color="orange", zorder=103)

            # plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            plot_landmarks(ax[1], config.LANDMARKS, color="black")
            # best = np.argmax(FlatParticle.w(particles))
            # covariances = FlatParticle.get_covariances(particles, best)
            # plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            # for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
            #     plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            for ii in range(config.N):
                plot_map(ax[1], FlatParticle.get_landmarks(particles, ii), color="orange", marker="o")
                covariances = FlatParticle.get_covariances(particles, ii)


            plt.pause(3)


        # if i == config.ODOMETRY.shape[0]-1:
        #     cuda.memcpy_dtoh(particles, memory.particles)
        #     best = np.argmax(FlatParticle.w(particles))
        #     best_covariances = FlatParticle.get_covariances(particles, best)
        #     best_landmarks = FlatParticle.get_landmarks(particles, best)


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        stats.stop_measuring("Loop")


    # if not plot:
    #     output = {
    #         "ground": [list(p) for p in stats.ground_truth_path],
    #         "predicted": [list(p) for p in stats.predicted_path],
    #         "landmarks": config.LANDMARKS.tolist(),
    #         "map": [list(lm) for lm in best_landmarks],
    #         "map_covariance": [cov.tolist() for cov in best_covariances]
    #     }

    #     fname = f"figs_fsonline/fixed_data_known_{config.N}_{config.THRESHOLD}_{seed}.json"

    #     with open(fname, "w") as f:
    #         json.dump(output, f)

    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     plot_history(ax, stats.ground_truth_path, color='green', markersize=1, linewidth=1, style="-", label="Ground truth")
    #     plot_history(ax, stats.predicted_path, color='orange', markersize=1, linewidth=1, style="-", label="SLAM estimate")
    #     # plot_history(ax, dead_reckoning, color='purple', markersize=1, linewidth=1, style="-", label="Dead reckoning")
    #     plot_landmarks(ax, config.LANDMARKS, color="blue")
    #     plot_map(ax, best_landmarks, color="orange", marker="o")
    #     for i, landmark in enumerate(best_landmarks):
    #         plot_confidence_ellipse(ax, landmark, best_covariances[i], n_std=3)

    #     plt.legend()
    #     fname = f"figs_fsonline/fixed_plot_known_{config.N}_{config.THRESHOLD}_{seed}.png"
    #     plt.savefig(fname)

    memory.free()
    stats.summary()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_fsonline_direct_known import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    run_SLAM(config, plot=False)