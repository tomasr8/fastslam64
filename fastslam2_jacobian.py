import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from particle3 import FlatParticle
import json
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit

from sensor import Sensor, wrap_angle
from stats import Stats
from common import CUDAMemory, resample, rescale, get_pose_estimate
from cuda.update2 import load_cuda_modules

def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        seed = config.SEED
    np.random.seed(seed)

    assert config.THREADS <= 1024  # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION.copy(), sigma=0.2)
    print("Particles memory:", particles.nbytes / 1024, "KB")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].axis('scaled')
        ax[1].axis('scaled')

    sensor = Sensor(
        config.LANDMARKS, [],
        config.sensor.VARIANCE, config.sensor.RANGE,
        config.sensor.FOV, config.sensor.MISS_PROB, 0, rb=True
    )

    # vehicle = Vehicle(config.START_POSITION, config.CONTROL_VARIANCE, dt=config.DT)
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

    cuda_modules["update"].get_function("init_rng")(
        np.int32(seed), block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
    )


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")

    pose = config.START_POSITION.copy()
    # dead_reckoning = [config.START_POSITION.copy()]

    for i in range(config.CONTROL.shape[0]):
        stats.start_measuring("Loop")
        print(i)

        # if i == 15:
        #     break

        # if i == 200:
        #     break

        stats.start_measuring("Measurement")
        # pose = config.GROUND_TRUTH[i]

        # ua = config.CONTROL[i, 0] + np.random.randn() * (config.CONTROL_VARIANCE[0] ** 0.5)
        # ub = config.CONTROL[i, 1] + np.random.randn() * (config.CONTROL_VARIANCE[1] ** 0.5)

        # dead_reckoning.append([
        #     dead_reckoning[-1][0] + ub * np.cos(dead_reckoning[-1][2]),
        #     dead_reckoning[-1][1] + ub * np.sin(dead_reckoning[-1][2]),
        #     wrap_angle(dead_reckoning[-1][2] + ua)
        # ])

        pose[0] += (config.DT * config.CONTROL[i, 1]) * np.cos(pose[2])
        pose[1] += (config.DT * config.CONTROL[i, 1]) * np.sin(pose[2])
        pose[2] += (config.DT * config.CONTROL[i, 0])
        pose[2] = wrap_angle(pose[2])
        # vehicle.move_noisy(config.CONTROL[i])


        measurements = sensor.get_noisy_measurements(pose)
        visible_measurements = measurements["observed"]
        missed_landmarks = measurements["missed"]
        out_of_range_landmarks = measurements["outOfRange"]

        print(len(visible_measurements))
        stats.stop_measuring("Measurement")


        # ===== RESET WEIGHTS =======
        cuda_modules["resample"].get_function("reset_weights")(
            memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda.memcpy_htod(memory.measurements, visible_measurements)

        if len(measurements) == 0:
            cuda_modules["predict"].get_function("predict_from_model")(
                memory.particles,
                np.float64(config.CONTROL[i, 0]), np.float64(config.CONTROL[i, 1]),
                np.float64(config.CONTROL_VARIANCE[0] ** 0.5), np.float64(config.CONTROL_VARIANCE[1] ** 0.5),
                np.float64(config.DT),
                block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
            )

        cuda_modules["update"].get_function("update")(
            memory.particles,
            memory.scratchpad, np.int32(memory.scratchpad_block_size),
            memory.measurements,
            np.int32(config.N), np.int32(len(visible_measurements)),
            memory.cov, np.float64(config.THRESHOLD),
            np.float64(config.sensor.RANGE), np.float64(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            np.float64(config.CONTROL[i, 0]), np.float64(config.CONTROL[i, 1]),
            np.float64(config.DT),
            block=(32, 1, 1), grid=(config.N//32, 1, 1)
        )
        
        rescale(cuda_modules, config, memory)
        estimate =  get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose(pose.tolist(), estimate.tolist())

        if plot:
            cuda.memcpy_dtoh(particles, memory.particles)

            visible_measurements = [[r*np.cos(b + pose[2]), r*np.sin(b + pose[2])]
                                    for (r, b) in visible_measurements]

            visible_measurements = np.array(visible_measurements)

            ax[0].clear()
            ax[1].clear()
            # ax[0].set_axis_off()
            # ax[1].set_axis_off()
            ax[0].axis('scaled')
            ax[1].axis('scaled')

            plot_sensor_fov(ax[0], pose, config.sensor.RANGE, config.sensor.FOV)
            plot_sensor_fov(ax[1], pose, config.sensor.RANGE, config.sensor.FOV)

            if(visible_measurements.size != 0):
                plot_connections(ax[0], pose, visible_measurements + pose[:2])

            plot_landmarks(ax[0], config.LANDMARKS, color="blue", zorder=100)
            plot_landmarks(ax[0], out_of_range_landmarks, color="black", zorder=101)
            plot_history(ax[0], stats.ground_truth_path, color='green')
            plot_history(ax[0], stats.predicted_path, color='orange')
            # plot_history(ax[0], config.DEAD_RECKONING[:(i+1)], color='purple')
            # plot_history(ax[0], dead_reckoning, color='purple')

            plot_particles_weight(ax[0], particles)
            if(visible_measurements.size != 0):
                plot_measurement(ax[0], pose[:2], visible_measurements, color="orange", zorder=103)

            plot_landmarks(ax[0], missed_landmarks, color="red", zorder=102)

            best = np.argmax(FlatParticle.w(particles))
            plot_landmarks(ax[1], config.LANDMARKS, color="black")
            covariances = FlatParticle.get_covariances(particles, best)

            plot_map(ax[1], FlatParticle.get_landmarks(particles, best), color="orange", marker="o")

            for i, landmark in enumerate(FlatParticle.get_landmarks(particles, best)):
                plot_confidence_ellipse(ax[1], landmark, covariances[i], n_std=3)

            plt.pause(0.001)


        # if i == config.CONTROL.shape[0]-1:
        #     cuda.memcpy_dtoh(particles, memory.particles)
        #     best = np.argmax(FlatParticle.w(particles))
        #     best_covariances = FlatParticle.get_covariances(particles, best)
        #     best_landmarks = FlatParticle.get_landmarks(particles, best)


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        # if i == config.CONTROL.shape[0]-1:
        #     cuda.memcpy_dtoh(particles, memory.particles)
        #     np.save("particles.npy", particles)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        # resample(cuda_modules, config, weights, memory, np.random.rand())

        stats.stop_measuring("Loop")


    # if not plot:
    #     output = {
    #         "ground": [list(p) for p in stats.ground_truth_path],
    #         "predicted": [list(p) for p in stats.predicted_path],
    #         "landmarks": config.LANDMARKS.tolist(),
    #         "map": [list(lm) for lm in best_landmarks],
    #         "map_covariance": [cov.tolist() for cov in best_covariances]
    #     }

    #     fname = f"figs_jacobi_dist/3_data_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}_{seed}.json"

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
    #     fname = f"figs_jacobi_dist/3_plot_{config.N}_{config.THRESHOLD}_{config.sensor.VARIANCE[0]:.2f}-{config.sensor.VARIANCE[1]:.4f}_{config.CONTROL_VARIANCE[0]:.4f}-{config.CONTROL_VARIANCE[1]:.2f}_{seed}.png"
    #     plt.savefig(fname)



    stats.summary()
    memory.free()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_jacobian_circle import config
    # from config_perf import config

    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)

    run_SLAM(config, plot=True)

    # context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    # print(repeat(run_SLAM, seeds=np.arange(100)))