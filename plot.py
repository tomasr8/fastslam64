import math
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

ground = np.load("ground.npy")
est = np.load("est.npy")

fig, ax = plt.subplots(2)

ax[0].plot(ground[:, 0], ground[0:, 1], c="green")
ax[0].plot(est[:, 0], est[0:, 1], c="orange")

ax[0].set_xlim(-10, 40)
ax[0].set_ylim(-10, 40)

ax[1].plot(np.arange(len(ground)), ground[:, 2], c="green")
ax[1].plot(np.arange(len(ground)), est[:, 2], c="orange")


# plt.axis("equal")

plt.show()