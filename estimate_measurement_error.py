import math
import numpy as np
from numpy.testing._private.utils import measure
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

T = 1000.0
ROBOT = 1
FOLDER = "utias32"

barcodes = np.load(f"{FOLDER}/barcodes.npy")
landmarks = np.load(f"{FOLDER}/landmarks.npy")
ground = np.load(f"{FOLDER}/npy/ground_{ROBOT}_50hz.npy")
measurements = np.load(f"{FOLDER}/npy/measurements_{ROBOT}_50hz.npy")

ground = ground[ground[:, 0] < T]
measurements = measurements[measurements[:, 0] < T]

true_measurements = []
noisy_measurements = []

for (t, x, y, theta) in ground:
    print(t)
    ms = measurements[measurements[:, 0] == t]

    for (_, barcode, r, b) in ms:
        noisy_measurements.append([r, b])

        # print(barcode)
        landmark_id = barcodes[barcodes[:, 1] == barcode][0, 0]
        landmark = landmarks[landmarks[:, 0] == landmark_id][0, 1:3]
        # print(landmark)
        lx, ly = landmark

        true_measurements.append([
            np.sqrt((x-lx)**2 + (y-ly)**2),
            pi_2_pi(np.arctan2(ly-y, lx-x) - theta)
        ])



true_measurements = np.array(true_measurements)
noisy_measurements = np.array(noisy_measurements)

residual = noisy_measurements - true_measurements

mu_est = np.mean(residual, axis=0)
print(mu_est)

cov_est = np.cov(residual.T)
print(cov_est)

print(f"Range: {np.sqrt(cov_est[0, 0])}**2")
print(f"Bearing: {np.rad2deg(np.sqrt(cov_est[1, 1]))}**2")


