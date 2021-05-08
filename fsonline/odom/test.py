import numpy as np

track = np.load("../track.npy")

track = np.vstack((
    track,
    np.array([
        [-1.4, 7.4],
        [-1.4, 6.3],
        [1.9, 7.4],
        [1.9, 6.3]
    ])
))

np.save("../track.npy", track)