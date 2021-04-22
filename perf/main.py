import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt


for N in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:

    landmarks = np.zeros((N, 2), dtype=np.float64)
    landmarks[:, 0] = np.linspace(0, 10, num=N)

    np.save(f"landmarks_{N}.npy", landmarks)

    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()