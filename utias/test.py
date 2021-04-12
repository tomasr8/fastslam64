import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

END = 1000.0

m1 = np.genfromtxt(f"csv/measurements_1_50hz.csv", delimiter=",")
m2 = np.load(f"npy/measurements_xy_1_50hz.npy")
m3 = np.load("npy_fixed/measurements_rb_1_50hz.npy")

m1 = m1[m1[:, 0] < END]
m2 = m2[m2[:, 0] < END]
m3 = m3[m3[:, 0] < END]


print(m1.shape)
print(m2.shape)
print(m3.shape)

length = m1[-1, 0] - m1[0, 0]

print(len(m1)/length)

print(m1[:10])
print(m2[:10])
print(m3[:10])


