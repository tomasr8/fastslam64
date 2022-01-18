import json

import numpy as np
import matplotlib.pyplot as plt
from lib.plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from lib.particle3 import FlatParticle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit
from lib.stats import Stats
from lib.common import CUDAMemory, resample, rescale, get_pose_estimate
from cuda.fastslam import load_cuda_modules
from lib.utils import number_to_color

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def rb2xy(pose, rb):
    [_, _, theta] = pose
    [r, b, color] = rb
    return [r * np.cos(b + theta), r * np.sin(b + theta), color]

def xy2rb(pose, landmark):
    color = landmark[2]
    position = pose[:2]
    vector_to_landmark = np.array(landmark[:2] - position, dtype=np.float64)

    r = np.linalg.norm(vector_to_landmark)
    b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]
    b = wrap_angle(b)

    return r, b, color


def run_SLAM(config, plot=False, seed=None, outpic="pic.png", outjson="out.json"):
    if seed is None:
        seed = config.SEED
    np.random.seed(seed)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
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
    stats.add_pose(config.START_POSITION, config.START_POSITION)

    plt.pause(1)

    for i in range(config.ODOMETRY.shape[0]):
        stats.start_measuring("Loop")
        print(i)

        stats.start_measuring("Measurement")

        pose = config.ODOMETRY[i]

        # colors = config.sensor.MEASUREMENTS[i][:, 2]
        visible_measurements = config.sensor.MEASUREMENTS[i]#[:, :2]
        visible_measurements = np.array([xy2rb(pose, m) for m in visible_measurements], dtype=np.float64)

        stats.stop_measuring("Measurement")

        cuda_modules["resample"].get_function("reset_weights")(
            memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda.memcpy_htod(memory.measurements, visible_measurements)

        cuda_modules["predict"].get_function("predict_from_imu")(
            memory.particles,
            np.float64(config.ODOMETRY[i, 0]), np.float64(config.ODOMETRY[i, 1]), np.float64(config.ODOMETRY[i, 2]),
            np.float64(config.ODOMETRY_VARIANCE[0] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[1] ** 0.5), np.float64(config.ODOMETRY_VARIANCE[2] ** 0.5),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        block_size = config.N if config.N < 32 else 32

        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(1),
            memory.scratchpad, np.int32(memory.scratchpad_block_size),
            memory.measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            memory.cov, np.float64(config.THRESHOLD),
            np.float64(config.sensor.RANGE), np.float64(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(block_size, 1, 1), grid=(config.N//block_size, 1, 1)
        )

        rescale(cuda_modules, config, memory)
        estimate = get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose([pose[0], pose[1], pose[2]], estimate)

        if plot:
            cuda.memcpy_dtoh(particles, memory.particles)

            ax[0].clear()
            ax[1].clear()
            ax[0].set_xlim([-20, 40])
            ax[0].set_ylim([-70, 30])
            ax[1].set_xlim([-20, 40])
            ax[1].set_ylim([-70, 30])

            major_x_ticks = np.arange(-20, 40, 20)
            minor_x_ticks = np.arange(-20, 40, 2)
            major_y_ticks = np.arange(-70, 30, 20)
            minor_y_ticks = np.arange(-70, 30, 2)

            ax[0].set_xticks(major_x_ticks)
            ax[0].set_xticks(minor_x_ticks, minor=True)
            ax[0].set_yticks(major_y_ticks)
            ax[0].set_yticks(minor_y_ticks, minor=True)
            ax[1].set_xticks(major_x_ticks)
            ax[1].set_xticks(minor_x_ticks, minor=True)
            ax[1].set_yticks(major_y_ticks)
            ax[1].set_yticks(minor_y_ticks, minor=True)

            ax[0].grid(which='minor', alpha=0.2)
            ax[0].grid(which='major', alpha=0.5)
            ax[1].grid(which='minor', alpha=0.2)
            ax[1].grid(which='major', alpha=0.5)

            plot_sensor_fov(ax[0], pose, config.sensor.RANGE, config.sensor.FOV)
            plot_sensor_fov(ax[1], pose, config.sensor.RANGE, config.sensor.FOV)

            visible_measurements = np.array([rb2xy(pose, m) for m in visible_measurements])

            if(visible_measurements.size != 0):
                plot_connections(ax[0], pose, visible_measurements[:, :2] + pose[:2])

            # plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
            plot_history(ax[0], stats.ground_truth_path, color='green')
            plot_history(ax[0], stats.predicted_path, color='orange')
            plot_history(ax[0], config.ODOMETRY[:i], color='red')

            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                cone_colors = [number_to_color(n) for n in visible_measurements[:, 2]]
                plot_measurement(ax[0], pose[:2], visible_measurements[:, :2], color=cone_colors, zorder=103)

            best = np.argmax(FlatParticle.w(particles))
            # plot_landmarks(ax[1], config.LANDMARKS, color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best),
                     [number_to_color(n) for n in FlatParticle.get_colors(particles, best)], marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            ax[0].arrow(pose[0], pose[1], 2.5*np.cos(pose[2]), 2.5*np.sin(pose[2]), color="green", width=0.2, head_width=0.5)

            plt.pause(0.001)


        if i == config.ODOMETRY.shape[0] - 1:
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


    n_landmarks = 0
    if not plot:
        output = {
            "map_size": len(best_landmarks),
            "ground": [list(v) for v in stats.ground_truth_path],
            "predicted": [list(v) for v in stats.predicted_path],
            # "landmarks": config.LANDMARKS.tolist(),
            "map": [list(lm) for lm in best_landmarks],
            "map_covariance": [cov.tolist() for cov in best_covariances]
        }

        with open(outjson, "w") as f:
            json.dump(output, f)

        fig, ax = plt.subplots()
        plot_history(ax, stats.ground_truth_path, color='green', linewidth=0.3, markersize=0.5)
        plot_history(ax, stats.predicted_path, color='orange', linewidth=0.3, markersize=0.5)
        # plot_landmarks(ax, config.LANDMARKS, color="blue")
        plot_map(ax, best_landmarks, color="orange", marker="o")
        for i, landmark in enumerate(best_landmarks):
            plot_confidence_ellipse(ax, landmark, best_covariances[i], n_std=3)

        plt.savefig(outpic)
        plt.close(fig)

        n_landmarks = len(best_landmarks)


    memory.free()
    stats.summary()
    return stats.mean_path_deviation(), n_landmarks


if __name__ == "__main__":
    from config_ins import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    run_SLAM(config, plot=True)
