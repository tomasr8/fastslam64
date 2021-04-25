import math
import numpy as np
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt


def get_particle_memory(n_landmarks):
    return 6 + 7*n_landmarks

def get_memory(n_particles, n_landmarks, known=False):
    memory = 0
    memory += n_particles * get_particle_memory(n_landmarks)
    memory += 2 * n_landmarks
    memory += 6 *n_particles

    if not known:
        memory += 2 * n_particles * n_landmarks

    memory *= 8

    return memory/(1024**2)


EXPORT = False

if EXPORT:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


fig, ax = plt.subplots()

fig.set_size_inches(w=5.02, h=3.0)
fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.99)

# particles = [1, 10, 100, 1000, 10000]
# n_landmarks = 1000
# memory = [get_memory(np, n_landmarks) for np in particles]
# print(memory)

n_particles = 128
landmarks = [1, 10, 100, 1000, 10000]
memory_unknown = [get_memory(n_particles, nl) for nl in landmarks]
memory_known = [get_memory(n_particles, nl, known=True) for nl in landmarks]

# print(memory)
ax.plot(landmarks, memory_unknown[:])
ax.plot(landmarks, memory_known[:])


ax.grid()
ax.set_ylabel("MB")
ax.set_xlabel("Number of landmarks")

# plt.xticks([])
# plt.yticks([])

# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
        #   fancybox=False, shadow=False, ncol=3)

if EXPORT:
    plt.savefig('memory.pgf')
else:
    plt.show()