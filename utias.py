import numpy as np
import matplotlib.pyplot as plt
from lib.plotting import (
    plot_connections, plot_history, plot_landmarks, plot_measurement,
    plot_particles_weight, plot_particles_grey, plot_confidence_ellipse,
    plot_sensor_fov, plot_map
)
from lib.particle3 import FlatParticle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.driver import limit
from cuda.fastslam import load_cuda_modules
from lib.stats import Stats
from lib.common import CUDAMemory, resample, rescale, get_pose_estimate

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def to_coords(range, bearing, theta):
    return [range * np.cos(bearing + theta), range * np.sin(bearing + theta)]

def run_SLAM(config, plot=False, seed=None):
    if seed is None:
        seed = config.SEED

    np.random.seed(seed)

    assert config.THREADS <= 1024 # cannot run more in a single block
    assert config.N >= config.THREADS
    assert config.N % config.THREADS == 0

    particles = FlatParticle.get_initial_particles(config.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2)

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

    for i, (g, o) in enumerate(zip(config.GROUND_TRUTH, config.CONTROL)):
        stats.start_measuring("Loop")

        stats.start_measuring("Measurement")

        t = g[0]
        measurements = config.sensor.MEASUREMENTS[config.sensor.MEASUREMENTS[:, 0] == t]
        measurements = measurements[:, [2,3]].astype(np.float64)

        stats.stop_measuring("Measurement")

        cuda.memcpy_htod(memory.measurements, measurements.copy())

        cuda_modules["resample"].get_function("reset_weights")(
            memory.particles,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )

        cuda_modules["predict"].get_function("predict_from_model")(
            memory.particles,
            np.float64(o[2]), np.float64(o[1]),
            np.float64(config.CONTROL_VARIANCE[0] ** 0.5), np.float64(config.CONTROL_VARIANCE[1] ** 0.5),
            np.float64(config.DT),
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )


        cuda_modules["update"].get_function("update")(
            memory.particles, np.int32(config.N//config.THREADS),
            memory.scratchpad, np.int32(memory.scratchpad_block_size),
            memory.measurements,
            np.int32(config.N), np.int32(len(measurements)),
            memory.cov, np.float64(config.THRESHOLD),
            np.float64(config.sensor.RANGE), np.float64(config.sensor.FOV),
            np.int32(config.MAX_LANDMARKS),
            block=(config.THREADS, 1, 1)
        )

        rescale(cuda_modules, config, memory)
        estimate =  get_pose_estimate(cuda_modules, config, memory)

        stats.add_pose(g[1:].tolist(), estimate.tolist())
        
        if plot and measurements.size > 0:
            cuda.memcpy_dtoh(particles, memory.particles)

            ax[0].clear()
            ax[1].clear()

            measurements = [to_coords(r, b, g[3]) for r, b in measurements]
            measurements = np.array(measurements)

            if(measurements.size != 0):
                plot_connections(ax[0], g[1:], measurements + g[1:3])

            plot_landmarks(ax[0], config.LANDMARKS[:, 1:], color="blue", zorder=100)
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


        cuda_modules["weights_and_mean"].get_function("get_weights")(
            memory.particles, memory.weights,
            block=(config.THREADS, 1, 1), grid=(config.N//config.THREADS, 1, 1)
        )
        cuda.memcpy_dtoh(weights, memory.weights)

        neff = FlatParticle.neff(weights)
        if neff < 0.6*config.N:
            resample(cuda_modules, config, weights, memory, 0.5)

        stats.stop_measuring("Loop")

    memory.free()
    stats.summary()
    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_utias import config
    context.set_limit(limit.MALLOC_HEAP_SIZE, config.GPU_HEAP_SIZE_BYTES)
    run_SLAM(config, plot=True)