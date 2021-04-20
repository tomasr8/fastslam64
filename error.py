import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import os

def relative_error(ground, predicted, indices):
    ground = np.array(ground)
    predicted = np.array(predicted)

    mse_trans = 0
    mse_rot = 0

    for i, j in indices:
        x = ground[j, 0] - ground[i, 0]
        y = ground[j, 1] - ground[i, 1]
        z = ground[j, 2] - ground[i, 2]

        a = predicted[j, 0] - predicted[i, 0]
        b = predicted[j, 1] - predicted[i, 1]
        c = predicted[j, 2] - predicted[i, 2]

        mse_trans += (x - a)**2 + (y - b)**2
        mse_rot += (z - c)**2

    mse_trans /= len(indices)
    mse_rot /= len(indices)

    return mse_trans, mse_rot


# def relative_error(ground, predicted):
#     ground = np.array(ground)
#     predicted = np.array(predicted)

#     true_deltas = []
#     predicted_deltas = []

#     for i in range(len(ground) -1):
#         true_deltas.append([
#             ground[i+1, 0] - ground[i, 0],
#             ground[i+1, 1] - ground[i, 1],
#             ground[i+1, 2] - ground[i, 2],
#         ])

#         predicted_deltas.append([
#             predicted[i+1, 0] - predicted[i, 0],
#             predicted[i+1, 1] - predicted[i, 1],
#             predicted[i+1, 2] - predicted[i, 2],
#         ])

#     rmse_trans = 0
#     rmse_rot = 0

#     for t, p in zip(true_deltas, predicted_deltas):
#         rmse_trans += (t[0] - p[0])**2 + (t[1] - p[1])**2
#         rmse_rot += (t[2] - p[2])**2

#     rmse_trans /= len(true_deltas)
#     rmse_trans = np.sqrt(rmse_trans)

#     rmse_rot /= len(true_deltas)
#     rmse_rot = np.sqrt(rmse_rot)

#     return rmse_trans, rmse_rot



for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    indices = [0, 123, 209, 313, 391, 478, 585, 683, 783, 915, 997, 1082, 1124]
    tuples = []
    for i in range(len(indices) - 1):
        tuples.append([indices[i], indices[i+1]])
    tuples.append([indices[-1], indices[0]])

    mse_trans = []
    mse_rot = []

    for filename in os.listdir('figs_jacobi_dist'):
        if filename.startswith(f"3_data_{N}") and filename.endswith(".json"):
            with open(f"figs_jacobi_dist/{filename}") as f:
                data = json.load(f)
                t, r = relative_error(data["ground"], data["predicted"], tuples)
                mse_trans.append(t)
                mse_rot.append(r)

        
    print(f"N: {N}")
    print(np.mean(mse_trans), np.std(mse_trans))
    print(np.mean(mse_rot), np.std(mse_rot))
    print()
