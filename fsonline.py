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

from sensor import Sensor
from stats import Stats

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def min_dist(arr, stamp):
    arr = np.abs(arr - stamp)

    # print(f"Min: {np.min(arr)}")

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

    if plot:
        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.axis('scaled')
        # ax[0].axis('scaled')
        # ax[1].axis('scaled')

    # sensor = Sensor(
    #     config.LANDMARKS, [],
    #     config.sensor.VARIANCE * 5, config.sensor.RANGE,
    #     config.sensor.FOV, config.sensor.MISS_PROB, 0, rb=True
    # )

    # stats = Stats("Loop", "Measurement")
    # stats.add_pose(config.START_POSITION, config.START_POSITION)
    # print("starting..")

    pose = np.copy(config.START_POSITION)

    all_measurements_real = []
    # all_measurements_fake = []


    for i in range(config.CONTROL.shape[0]):
        # stats.start_measuring("Loop")
        # print(f"iteration: {i}")


        # stats.start_measuring("Measurement")
        stamp = config.ODOMETRY[i, -1]

        pose = motion_model(pose, config.CONTROL[i])

        # print(pose, config.ODOMETRY[i, :-1])

        # if i % 15 == 0:
        #     measurements = sensor.get_noisy_measurements(pose)
        #     visible_measurements = measurements["observed"]
        # else:
        #     visible_measurements = np.zeros(0, dtype=np.float64)

        # for m in visible_measurements:
        #     rel = rb2xy(pose, m)
        #     all_measurements_fake.append([pose[0] + rel[0], pose[1] + rel[1]])

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

            # for i in range(len(measurements)):
            #     measurements[i] = xy2rb(pose, measurements[i])

            # measurements[:, 0] -= pose[0]
            # measurements[:, 1] -= pose[1]

            visible_measurements = np.copy(measurements).astype(np.float64)


        for m in visible_measurements:
            # rel = rb2xy(pose, m)
            # all_measurements_real.append([pose[0] + rel[0], pose[1] + rel[1]])
            all_measurements_real.append(m)


        # measured_pose = [
        #     pose[0] + np.random.normal(0, config.ODOMETRY_VARIANCE[0]),
        #     pose[1] + np.random.normal(0, config.ODOMETRY_VARIANCE[1]),
        #     pose[2] + np.random.normal(0, config.ODOMETRY_VARIANCE[2])
        # ]

        # stats.stop_measuring("Measurement")


        if False:#plot:
            # ax[0].clear()
            # ax[1].clear()
            # ax[0].set_xlim([-160, 10])
            # ax[0].set_ylim([-30, 50])
            # ax[1].set_xlim([-160, 10])
            # ax[1].set_ylim([-30, 50])
            # ax[0].set_axis_off()
            # ax[1].set_axis_off()

            ax.clear()
            ax.set_xlim([-160, 10])
            ax.set_ylim([-30, 50])
            ax.set_axis_off()


            visible_measurements = np.array([rb2xy(pose, m) for m in visible_measurements])

            # all_measurements = np.array(all_measurements)

            # if(visible_measurements.size != 0):
                # plot_connections(ax[0], pose, visible_measurements + pose[:2])
                # plot_measurement(ax[0], pose[:2], visible_measurements, color="orange", zorder=103)

            # print(np.array(all_measurements))
            if len(all_measurements_real) > 0:
                plot_measurement(ax, np.zeros(2), np.array(all_measurements_real), color="orange", zorder=103, size=3)

            # if len(all_measurements_fake) > 0:
            #     plot_measurement(ax[1], np.zeros(2), np.array(all_measurements_fake), color="orange", zorder=103, size=5)

            plot_landmarks(ax, config.LANDMARKS, color="blue", zorder=104, s=4)
            # plot_landmarks(ax[1], config.LANDMARKS, color="blue", zorder=100)

            # plot_history(ax[0], stats.ground_truth_path, color='green')
            # plot_history(ax[0], stats.predicted_path, color='orange')
            plot_history(ax, config.ODOMETRY[:i], color='red')


            plt.pause(0.01)
            # if i > 20:
            #     plt.pause(2)


        # stats.stop_measuring("Loop")


    ax.clear()
    ax.set_xlim([-160, 10])
    ax.set_ylim([-30, 50])
    ax.set_axis_off()

    plot_measurement(ax, np.zeros(2), np.array(all_measurements_real), color="orange", zorder=103, size=3)

    plot_landmarks(ax, config.LANDMARKS, color="blue", zorder=104, s=4)
    plot_history(ax, config.ODOMETRY[:i], color='red')


    plt.show()

    return []


if __name__ == "__main__":
    from config_jacobian_fsonline import config
    run_SLAM(config, plot=True)