import math
import json
import numpy as np
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from plotting import plot_history, plot_measurement, plot_landmarks

EXPORT = False

if EXPORT:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


# with open("figs_utias/histogram_3_1.json") as f:
#     histogram = json.load(f)

with open("figs_utias/2_histogram.json") as f:
    histogram = json.load(f)

fig, ax = plt.subplots()

fig.set_size_inches(w=5.02, h=5.02)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.99)

ground = np.array(histogram["ground"])
# measurements = np.array(histogram["measurements"])
# landmarks = np.array(histogram["landmarks"])

plot_history(ax, ground[:5000:50], color='green', linewidth=1, label="Robot path")
plot_history(ax, ground[-1000::50], color='orange', linewidth=1, label="Robot path")
# plot_landmarks(ax, landmarks, color="blue", zorder=104, label="Landmarks")
# plot_measurement(ax, [0, 0], measurements, color="orange", zorder=103, size=3, label="Measurements")

plt.xticks([])
plt.yticks([])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
          fancybox=False, shadow=False, ncol=3)

if EXPORT:
    plt.savefig('histogram.pgf')
else:
    plt.show()