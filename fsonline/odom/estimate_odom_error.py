import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def process_compare(data):
    true_odom = []

    for i in range(len(data)):
        x, y, theta = data[i]["true_odom"]
        theta = pi_2_pi(theta)
        true_odom.append([x, y, theta])


    true_odom = np.array(true_odom)
    true_odom[:, [0, 1, 2]] = true_odom[:, [1, 0, 2]]
    true_odom[:, 1] +=2
    true_odom[:, 0] *= -1
    true_odom[:, 2] += np.pi/2
    true_odom[:, 2] = pi_2_pi(true_odom[:, 2])

    return true_odom


xavier_odom = np.load("fixed_odom.npy")

with open("odom_measurements_compare.json") as f:
    data = json.load(f)

true_odom = process_compare(data)


residual = xavier_odom - true_odom

mu_est = np.mean(residual, axis=0)
print(mu_est)

cov_est = np.cov(residual.T)
print(cov_est)

print(f"X: {np.sqrt(cov_est[0, 0])}**2")
print(f"Y: {np.sqrt(cov_est[1, 1])}**2")
print(f"Theta: {np.rad2deg(np.sqrt(cov_est[2, 2]))}**2")
print(f"Theta: {np.sqrt(cov_est[2, 2])}**2")

