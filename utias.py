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

from stats import Stats

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

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        fig.tight_layout()


    stats = Stats("Loop", "Measurement")
    stats.add_pose(config.START_POSITION.tolist(), config.START_POSITION.tolist())
    print("starting..")

    all_measurements = []

    for i, (g, o) in enumerate(zip(config.GROUND_TRUTH, config.CONTROL)):
        stats.start_measuring("Loop")
        print(f"{i}/{config.CONTROL.shape[0]}")


        stats.start_measuring("Measurement")

        t = g[0]
        measurements = config.sensor.MEASUREMENTS[config.sensor.MEASUREMENTS[:, 0] == t]
        measurements = measurements[:, [2,3]].astype(np.float64)

        stats.stop_measuring("Measurement")

        stats.add_pose(g[1:].tolist(), [0, 0, 0])

        measurements = [to_coords(r, b, g[3]) for r, b in measurements]
        measurements = np.array(measurements)

        if measurements.size > 0:
            for m in measurements:
                all_measurements.append([m[0] + g[1], m[1] + g[2]])

        if plot and measurements.size > 0:# and i % 10 == 0:
            ax[0].clear()
            ax[1].clear()

            ax[0].text(0, 5.5, f"Iteration: {i}")

            # all_measurements = np.array(all_measurements)

            if(measurements.size != 0):
                plot_connections(ax[0], g[1:], measurements + g[1:3])

            plot_history(ax[0], stats.ground_truth_path[::50], color='green')

            if(len(all_measurements) >  0):
                plot_measurement(ax[0], [0, 0], np.array(all_measurements), color="orange", zorder=103)

            plot_landmarks(ax[0], config.LANDMARKS[:, 1:], color="blue", zorder=104)

            plt.pause(0.001)


        stats.stop_measuring("Loop")

    stats.summary()

    if not plot:
        output = {
            "ground": stats.ground_truth_path,
            "measurements": all_measurements,
            "landmarks": config.LANDMARKS[:, 1:].tolist()
        }

        with open(f"figs_utias/histogram_{config.DATASET}_{config.ROBOT}.json", "w") as f:
            json.dump(output, f)

        # fig, ax = plt.subplots()
        # plot_history(ax, stats.ground_truth_path[::50], color='green')
        # plot_measurement(ax, [0, 0], np.array(all_measurements), color="orange", zorder=103, size=3)
        # plot_landmarks(ax, config.LANDMARKS[:, 1:], color="blue", zorder=104)

        # plt.savefig(f"figs_utias/histogram_{config.ROBOT}_{seed}.png")

    return stats.mean_path_deviation()


if __name__ == "__main__":
    from config_utias import config
    run_SLAM(config, plot=False)