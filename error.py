import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import os

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

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
        mse_rot += wrap_angle(z - c)**2

    mse_trans /= len(indices)
    mse_rot /= len(indices)

    return mse_trans, mse_rot


def mse(ground, predicted):
    ground = np.array(ground)
    predicted = np.array(predicted)

    mse_trans = 0
    mse_rot = 0

    for g, p in zip(ground, predicted):
        mse_trans += (g[0]-p[0])**2 + (g[1]-p[1])**2
        mse_rot += wrap_angle(g[2]-p[2])**2

    N = len(ground)

    return mse_trans/N, mse_rot/N


def run_rel():
    # for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:

        # indices = [0, 123, 209, 313, 391, 478, 585, 683, 783, 915, 997, 1082, 1124]
        # tuples = []
        # for i in range(len(indices) - 1):
        #     tuples.append([indices[i], indices[i+1]])
        # tuples.append([indices[-1], indices[0]])

        tuples = [
            (0, 30000), (30000, 66660), (0, 15000), (15000, 45000),
            (45000, 66660), (0, 66660), (0, 10000), (10000, 20000),
            (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000),
            (60000, 66660)
        ]

        rel_trans = []
        rel_rot = []

        # for filename in os.listdir('figs_utias'):
        #     if filename.startswith(f"2_data_3_1_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_utias/{filename}") as f:
        for filename in os.listdir('figs_utias'):
            if filename.startswith(f"2_known_data_3_1_{N}_") and filename.endswith(".json"):
                with open(f"figs_utias/{filename}") as f:
        # for filename in os.listdir('figs_jacobi_dist'):
        #     if filename.startswith(f"2_known_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_jacobi_dist/{filename}") as f:
        # for filename in os.listdir('figs_jacobi_dist'):
        #     if filename.startswith(f"3_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_jacobi_dist/{filename}") as f:
                    data = json.load(f)
                    t, r = relative_error(data["ground"], data["predicted"], tuples)
                    rel_trans.append(t)
                    rel_rot.append(r)

            
        print(f"N: {N}")
        print(np.mean(rel_trans), np.std(rel_trans))
        print(np.mean(rel_rot), np.std(rel_rot))
        print()


def run_mse():
    # for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    # for N in [4, 8, 16, 32, 64, 128]:
    # for N in [2048, 4096, 8192]:

        mse_trans = []
        mse_rot = []

        # for filename in os.listdir('figs_pr'):
        #     if filename.startswith(f"1_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_pr/{filename}") as f:
        # for filename in os.listdir('figs_jacobi_dist'):
        #     if filename.startswith(f"1_known_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_jacobi_dist/{filename}") as f:
        # for filename in os.listdir('figs_jacobi_dist'):
        #     if filename.startswith(f"3_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_jacobi_dist/{filename}") as f:
        # for filename in os.listdir('figs_jacobi_dist'):
        #     if filename.startswith(f"2_known_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_jacobi_dist/{filename}") as f:
        # for filename in os.listdir('figs_utias'):
        #     if filename.startswith(f"2_data_3_1_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_utias/{filename}") as f:
        # for filename in os.listdir('figs_utias'):
        #     if filename.startswith(f"2_known_data_3_1_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_utias/{filename}") as f:
        # for filename in os.listdir('figs_fsonline'):
        #     if filename.startswith(f"1_data_{N}_") and filename.endswith(".json"):
        #         with open(f"figs_fsonline/{filename}") as f:
        # for filename in os.listdir('figs_fsonline'):
        #     if filename.startswith(f"fixed_data_{N}_") and filename.endswith(".json"):
        for filename in os.listdir('figs_fsonline'):
            if filename.startswith(f"fixed_data_known_{N}_") and filename.endswith(".json"):
                with open(f"figs_fsonline/{filename}") as f:
                    data = json.load(f)
                    t, r = mse(data["ground"], data["predicted"])
                    mse_trans.append(t)
                    mse_rot.append(r)

            
        print(f"N: {N}")
        print(np.mean(mse_trans), np.std(mse_trans))
        print(np.mean(mse_rot), np.std(mse_rot))
        print()


run_mse()
# run_rel()
