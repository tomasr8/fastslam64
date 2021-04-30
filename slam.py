import math
import json
import numpy as np
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

from plotting import plot_history, plot_measurement, plot_landmarks, plot_confidence_ellipse

def mean_path_deviation(ground, predicted):
    ground_truth = np.array(ground)[:, :2]
    predicted = np.array(predicted)[:, :2]

    dist = np.linalg.norm(ground_truth - predicted, axis=1)**2
    return np.sqrt(np.mean(dist))

def relative_error(ground, predicted):
    ground = np.array(ground)[:, :2]
    predicted = np.array(predicted)[:, :2]

    true_deltas = []
    predicted_deltas = []

    for i in range(len(ground) -1):
        true_deltas.append([
            ground[i+1, 0] - ground[i, 0],
            ground[i+1, 1] - ground[i, 1]
        ])

        predicted_deltas.append([
            predicted[i+1, 0] - predicted[i, 0],
            predicted[i+1, 1] - predicted[i, 1]
        ])

    rmse = 0

    for t, p in zip(true_deltas, predicted_deltas):
        rmse += (t[0] - p[0])**2 + (t[1] - p[1])**2

    rmse /= len(true_deltas)
    rmse = np.sqrt(rmse)

    return rmse



EXPORT = True

if EXPORT:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


# with open("figs_jacobi_dist/3_data_1024_0.8_0.02-0.0003_0.0076-0.02_16.json") as f:
#     data = json.load(f)
with open("figs_utias/2_data_3_1_8192_1.3_0.18-0.0024_0.0195-0.01_5.json") as f:
    data = json.load(f)


fig, ax = plt.subplots()

fig.set_size_inches(w=5.02, h=5.5)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.99)

ground = np.array(data["ground"])[::50]
predicted = np.array(data["predicted"])[::50]
# dr = np.array(data["dead_reckoning"])
landmarks = np.array(data["landmarks"])
estimated_landmarks = np.array(data["map"])
covariance = np.array(data["map_covariance"])

# plot_history(ax, ground[::50], color='green', linewidth=1, label="Robot path")
# plot_history(ax, dr[::50], color='purple', linewidth=1, style="--", label="Dead reckoning")
# plot_landmarks(ax, landmarks, color="blue", zorder=104, label="Landmarks")

plot_history(ax, ground, color='green', linewidth=1, label="Robot path")
plot_history(ax, predicted, color='orange', linewidth=1, label="Estimated path")
plot_landmarks(ax, landmarks, color="blue", zorder=104, label="Landmarks")
plot_landmarks(ax, estimated_landmarks, color="orange", zorder=104, label="Estimated landmarks")

# for i, landmark in enumerate(estimated_landmarks):
#     plot_confidence_ellipse(ax, landmark, covariance[i], n_std=3, zorder=105)

# plot_measurement(ax, [0, 0], measurements, color="orange", zorder=103, size=3, label="Measurements")

plt.xticks([])
plt.yticks([])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0),
          fancybox=False, shadow=False, ncol=2, columnspacing=0.5)

print(mean_path_deviation(ground, predicted))
print(relative_error(ground, predicted))


if EXPORT:
    plt.savefig('utias_vis_est.pgf')
else:
    plt.show()