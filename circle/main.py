import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

np.random.seed(7)
N = 20

L1 = np.random.rand(N//4, 2) * 5
L2 = np.random.rand(N//4, 2) * 5
L2[:, 0] += 5
L3 = np.random.rand(N//4, 2) * 5
L3[:, 1] += 5
L4 = np.random.rand(N//4, 2) * 5
L4 += 5

landmarks = np.vstack((
    L1, L2, L3, L4
))

np.save("landmarks.npy", landmarks)

plt.scatter(landmarks[:, 0], landmarks[:, 1])
plt.show()