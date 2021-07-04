import time
import numpy as np

class Event(object):
    def __init__(self, name):
        self.name = name
        self.times = []
        self.start = None

    def start_measuring(self):
        self.start = time.time()

    def stop_measuring(self):
        if self.start is None:
            raise Exception("start_measuring() was not called")

        self.times.append(time.time() - self.start)

class Stats(object):
    def __init__(self, *events):
        self.events = {
            name: Event(name) for name in events
        }

        self.ground_truth_path = []
        self.predicted_path = []

    def start_measuring(self, name):
        self.events[name].start_measuring()

    def stop_measuring(self, name):
        self.events[name].stop_measuring()

    def add_pose(self, ground_truth, predicted):
        self.ground_truth_path.append(ground_truth)
        self.predicted_path.append(predicted)

    def mean_path_deviation(self):
        ground_truth = np.array(self.ground_truth_path)[:, :2]
        predicted = np.array(self.predicted_path)[:, :2]

        dist = np.linalg.norm(ground_truth - predicted, axis=1)
        return np.mean(dist), np.std(dist)

    def summary(self, names=None):
        if names is None:
            names = self.events.keys()

        for name in names:
            event = self.events[name]
            times = np.array(event.times) * 1000
            mean = np.mean(times)
            std = np.std(times)

            print(f"<{name}> average: {mean:.1f}ms | std: {std:.1f}")

        mean, std = self.mean_path_deviation()
        print(f"<Mean path deviation> {mean:.2f} | std: {std:.2f}")