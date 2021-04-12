import math
import time
import numpy as np
from numpy.testing._private.utils import measure
import scipy
import scipy.stats
import matplotlib.pyplot as plt

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
from sensor import Sensor
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def min_dist(arr, stamp):
    arr = np.abs(arr - stamp)

    print(f"Min: {np.min(arr)}")

    argmin = np.argmin(arr)
    if arr[argmin] < 30:
        return argmin
    else:
        return None

def xy2rb(pose, landmark):
    position = pose[:2]
    vector_to_landmark = np.array(landmark - position, dtype=np.float64)

    r = np.linalg.norm(vector_to_landmark)
    b = np.arctan2(vector_to_landmark[1], vector_to_landmark[0]) - pose[2]
    b = wrap_angle(b)

    return r, b

def rb2xy(pose, rb):
    [_, _, theta] = pose
    [r, b] = rb
    return [r * np.cos(b + theta), r * np.sin(b + theta)]


def motion_model(pose, u):
    [x, y, theta] = pose

    theta += u[0]
    theta = wrap_angle(theta)

    x += np.cos(theta) * u[1]
    y += np.sin(theta) * u[1]

    theta += u[2]
    theta = wrap_angle(theta)

    return [x, y, theta]


def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        seed = config.SEED
    np.random.seed(seed)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    stamps = np.array([m['stamp'] for m in config.sensor.MEASUREMENTS])

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].axis('scaled')
        ax[1].axis('scaled')

    cuda_modules = config.modules

    sensor = Sensor(
        config.LANDMARKS, [],
        config.sensor.VARIANCE * 5, config.sensor.RANGE,
        config.sensor.FOV, config.sensor.MISS_PROB, 0, rb=True
    )

    memory = CUDAMemory(config)
    weights = np.zeros(config.N, dtype=np.float64)

    cuda.memcpy_htod(memory.cov, config.sensor.COVARIANCE)
    cuda.memcpy_htod(memory.particles, particles)

    cuda_modules["predict"].get_function("init_rng")(
        np.int32(config.SEED), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION, config.START_POSITION)
    print("starting..")

    pose = np.copy(config.START_POSITION)
    dead_reckoning = [np.copy(config.START_POSITION)]

    for i in range(config.CONTROL.shape[0]):
        stats.start_measuring("Loop")
        print(f"iteration: {i}")

        # if i == 80:
        #     print("savving")
        #     np.save("ground.npy", stats.ground_truth_path)
        #     np.save("est.npy", stats.predicted_path)

        stats.start_measuring("Measurement")
        stamp = config.ODOMETRY[i, -1]
        # pose = config.ODOMETRY[i, :3]

        # ua = config.CONTROL[i, 0] + np.random.randn() * (config.CONTROL_VARIANCE[0] ** 0.5)
        # ub = config.CONTROL[i, 1] + np.random.randn() * (config.CONTROL_VARIANCE[1] ** 0.5)

        # dead_reckoning.append([
        #     dead_reckoning[-1][0] + ub * np.cos(dead_reckoning[-1][2]),
        #     dead_reckoning[-1][1] + ub * np.sin(dead_reckoning[-1][2]),
        #     wrap_angle(dead_reckoning[-1][2] + ua)
        # ])
        dead_reckoning.append(motion_model(dead_reckoning[-1], config.CONTROL[i]))

        # pose[0] += (config.DT * config.CONTROL[i, 1]) * np.cos(pose[2])
        # pose[1] += (config.DT * config.CONTROL[i, 1]) * np.sin(pose[2])
        # pose[2] += (config.DT * config.CONTROL[i, 0])
        # pose[2] = wrap_angle(pose[2])

        pose = motion_model(pose, config.CONTROL[i])

        print(pose, config.ODOMETRY[i, :-1])

        if i < 15:
            measurements = sensor.get_noisy_measurements(pose)
            fake_measurements = measurements["observed"]
        else:
            fake_measurements = np.zeros(0, dtype=np.float64)

        argmin = min_dist(stamps, stamp)
        if argmin is None:
            visible_measurements = np.array([], dtype=np.float64)
        else:
            measurements = config.sensor.MEASUREMENTS[argmin]
            measurements = np.array(measurements['points'])[:, :2]
            measurements[:, [0, 1]] = measurements[:, [1, 0]]
            measurements[:, 1] +=2
            measurements[:, 0] *= -1

            # measurements[:, 1] -= 0.3

            for j in range(len(measurements)):
                measurements[j] = xy2rb(pose, measurements[j])

            # measurements[:, 0] -= pose[0]
            # measurements[:, 1] -= pose[1]

            visible_measurements = np.copy(measurements).astype(np.float64)

        if len(fake_measurements) > 0:
            print("Adding fake measurements")
            if len(visible_measurements) == 0:
                visible_measurements = fake_measurements
            else:
                visible_measurements = np.vstack((visible_measurements, fake_measurements))

        # measured_pose = [
        #     pose[0] + np.random.normal(0, config.ODOMETRY_VARIANCE[0]),
        #     pose[1] + np.random.normal(0, config.ODOMETRY_VARIANCE[1]),
        #     pose[2] + np.random.normal(0, config.ODOMETRY_VARIANCE[2])
        # ]

        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(memory.measurements, visible_measurements)

        cuda_modules["predict"].get_function("predict_from_fsonline_model")(
            memory.particles,
            np.float64(config.CONTROL[i, 0]), np.float64(config.CONTROL[i, 1]), np.float64(config.CONTROL[i, 2]),
            np.float64(config.CONTROL_VARIANCE[0] ** 0.5), np.float64(config.CONTROL_VARIANCE[1] ** 0.5),
            np.float64(config.DT),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )


        # if plot:
        #     cuda.memcpy_dtoh(particles, memory.particles)

        #     ax[0].clear()
        #     ax[1].clear()
        #     ax[0].set_xlim([-160, 10])
        #     ax[0].set_ylim([-30, 50])
        #     ax[1].set_xlim([-160, 10])
        #     ax[1].set_ylim([-30, 50])
        #     ax[0].set_axis_off()
        #     ax[1].set_axis_off()

        #     plot_history(ax[0], stats.ground_truth_path, color='green')
        #     plot_history(ax[0], stats.predicted_path, color='orange')

        #     plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
        #     plot_history(ax[0], stats.ground_truth_path, color='green')
        #     plot_history(ax[0], stats.predicted_path, color='orange')

        #     plot_particles_weight(ax[0], particles)
        #     plt.pause(0.2)


        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(config.N//config.THREADS),
            memory.scratchpad, np.int32(memory.scratchpad_block_size),
            memory.measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            memory.cov, np.float64(config.THRESHOLD),
            np.float64(config.sensor.RANGE), np.float64(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )

        # cuda.memcpy_dtoh(particles, memory.particles)
        # print(f"Particle weights before rescale: {FlatParticle.w(particles)}")
        rescale(cuda_modules, config, memory)
        estimate = get_pose_estimate(cuda_modules, config, memory)

        # cuda.memcpy_dtoh(particles, memory.particles)
        # print(f"Particle weights after rescale: {FlatParticle.w(particles)}")
        # print(f"Particle xs: {FlatParticle.x(particles)}")
        # print(f"Particle ys: {FlatParticle.y(particles)}")

        print(f"Estimate: {estimate}, Groud: {pose}")

        stats.add_pose([pose[0], pose[1], pose[2]], estimate)

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

            visible_measurements = np.array([rb2xy(pose, m) for m in visible_measurements])

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

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], config.LANDMARKS, color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)
            # if i > 20:
            #     plt.pause(2)


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


    if not plot:
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_history(ax, stats.ground_truth_path, color='green', markersize=1, linewidth=1, style="-", label="Ground truth")
        plot_history(ax, stats.predicted_path, color='orange', markersize=1, linewidth=1, style="-", label="SLAM estimate")
        plot_history(ax, dead_reckoning, color='purple', markersize=1, linewidth=1, style="-", label="Dead reckoning")
        plot_landmarks(ax, config.LANDMARKS, color="blue")
        plot_map(ax, best_landmarks, color="orange", marker="o")
        for i, landmark in enumerate(best_landmarks):
            plot_confidence_ellipse(ax, landmark, best_covariances[i], n_std=3)

        plt.legend()
        plt.savefig(f"figs_fsonline/{seed}.png")
        print(f"figs_fsonline/{seed}.png")

    stats.summary()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_jacobian_fsonline import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    run_SLAM(config, plot=False)