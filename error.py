import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import os

def relative_error(ground, predicted):
    ground = np.array(ground)[:, :2]
    predicted = np.array(predicted)[:, :2]

    true_deltas = []
    predicted_deltas = []

    for i in range(len(ground) -1):
        true_deltas.append([
            ground[i+1, 0] - ground[i, 0],
            ground[i+1, 1] - ground[i, 1]
        ])

        predicted_deltas.append([
            predicted[i+1, 0] - predicted[i, 0],
            predicted[i+1, 1] - predicted[i, 1]
        ])

    rmse = 0

    for t, p in zip(true_deltas, predicted_deltas):
        rmse += (t[0] - p[0])**2 + (t[1] - p[1])**2

    rmse /= len(true_deltas)
    rmse = np.sqrt(rmse)

    return rmse




for filename in os.listdir('figs_fsonline'):
    if filename.startswith("5data") and filename.endswith(".json"):
        print(filename)
