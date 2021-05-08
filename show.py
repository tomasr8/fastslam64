import math
import numpy as np
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

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
fig.set_size_inches(w=5.02, h=4)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.99)

# UTIAS
# =====================================================
# error = [4.10, 2.08, 1.18, 2.00, 2.40, 1.02, 0.60, 0.82, 0.30, 0.06, 0.03, 0.03]
# yerr = [0.41, 0.17, 0.08, 0.14, 0.15, 0.09, 0.05, 0.07, 0.02, 0.02, 0.02, 0.02]
# ax.errorbar(np.arange(12), error, yerr=yerr, capsize=5, label="Unknown correspondence")

# error = [0.11, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
# yerr = [0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
# ax.errorbar(np.arange(12), error, yerr=yerr, capsize=5, label="Known correspondence")

# # error = [5.73, 2.83, 2.25, 2.47, 2.78, 1.42, 1.03, 1.16, 0.44, 0.30, 0.03, 0.03]
# # yerr = [6.28, 3.52, 1.95, 3.24, 2.96, 2.14, 1.96, 2.09, 1.12, 0.93, 0.01, 0.01]
# # ax.errorbar(np.arange(12), error, yerr=yerr, capsize=5, label="Rel(trans) unknown correspondence")

# # error = [0.12, 0.06, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
# # yerr = [0.04, 0.02 ,0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# # ax.errorbar(np.arange(12), error, yerr=yerr, capsize=5, label="Rel(trans) known correspondence")

# plt.xticks(ticks=np.arange(12), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], rotation = 45)
# ax.set_ylabel("MSE(trans)")
# plt.legend()
# =====================================================


# FS ONLINE
# =====================================================
error = [0.18, 0.16, 0.12, 0.11, 0.10, 0.09, 0.06, 0.05, 0.05]
yerr = [0.01, 0.02, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01, 0.00]
ax.errorbar(np.arange(9), error, yerr=yerr, capsize=5, label="Unknown correspondence")

plt.xticks(ticks=np.arange(9), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024], rotation = 45)
ax.set_ylabel("MSE(trans)")
plt.legend()
# =====================================================



# square
# error = [3.89, 1.9, 1.42, 0.84, 0.65, 0.6, 0.54, 0.75, 0.37]
# yerr = [2.79, 1.74, 1.08, 0.56, 0.48, 0.54, 0.41, 1.53, 0.3]

# plt.errorbar(np.arange(9), error, yerr=yerr, capsize=5)
# plt.xticks(ticks=np.arange(9), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024])

# circle
# =====================================================
# error = [11.73, 5.6, 3.12, 1.86, 0.52, 0.15, 0.06, 0.07, 0.08]
# yerr = [7, 5.96, 3.95, 2.33, 1.56, 0.29, 0.05, 0.08, 0.07]
# ax.errorbar(np.arange(9), error, yerr=yerr, capsize=5, label="Ours (unknown correspondence)")

# error = [0.35, 0.18, 0.14, 0.12, 0.07, 0.08, 0.09, 0.05, 0.06]
# yerr = [0.28, 0.14, 0.11, 0.11, 0.04, 0.08, 0.06, 0.03, 0.05]
# ax.errorbar(np.arange(9), error, yerr=yerr, capsize=5, label="Ours (known correspondence)")

# error = [1.59, 1.08, 1.4, 0.82, 0.98, 0.88]
# yerr = [1.52, 0.6, 1.5, 0.6, 0.92, 0.97]
# ax.errorbar(np.arange(6), error, yerr=yerr, capsize=5, label="Python Robotics (known correspondence)")
# plt.xticks(ticks=np.arange(9), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
# ax.set_ylabel("MSE(trans)")
# plt.legend()
# =====================================================

# 3_circle relative error
# error = [7.32, 3.72, 2.1, 1.64, 0.47, 0.1, 0.06, 0.06, 0.06]
# yerr = [5.1, 5.88, 2.32, 1.86, 1.44, 0.16, 0.05, 0.05, 0.06]

# plt.errorbar(np.arange(9), error, yerr=yerr, capsize=5)
# plt.xticks(ticks=np.arange(9), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024])


# python robotics time
# time_pr = [318, 162, 80, 43, 19, 11, 5, 3, 1]
# time_mine = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 600]

# plt.plot(np.arange(9), time_pr)
# plt.plot(np.arange(9), time_mine)
# plt.xticks(ticks=np.arange(9), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024])

# perf simult measurements
# ================================================
# error = [0.18, 0.32, 0.64, 1.02, 3.26, 14.86, 124.2]
# yerr = [0.00, 0.01, 0.01, 0.02, 0.02, 0.09, 0.04]
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Unknown correspondence (128)")

# error = [0.11, 0.25, 0.41, 0.81, 1.55, 2.91, 4.92]
# yerr = [0.0, 0, 0.0, 0.01, 0.00, 0.01, 0.01]
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Known correspondence (128)")

# error = [0.21, 0.35, 0.72, 1.29, 4.1, 20.8, 158]
# yerr = [0.01, 0.01, 0.01, 0.02, 0.16, 0.12, 0.47]
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Unknown correspondence (1024)")

# error = [0.15, 0.36, 0.66, 1.28, 2.48, 4.68, 8.9]
# yerr = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0]
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Known correspondence (1024)")

# plt.xticks(ticks=[4, 16, 32, 64, 128, 256], rotation=45)
# ax.set_ylabel("Time (s)")
# plt.legend()
# ================================================


# HZ perf simult measurements
# ================================================
# error = 200/np.array([0.18, 0.32, 0.64, 1.02, 3.26, 14.86, 124.2])
# yerr = np.array([0.00, 0.01, 0.01, 0.02, 0.02, 0.09, 0.04])
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Unknown correspondence (128)")

# error = 200/np.array([0.11, 0.25, 0.41, 0.81, 1.55, 2.91, 4.92])
# yerr = np.array([0.0, 0, 0.0, 0.01, 0.00, 0.01, 0.01])
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Known correspondence (128)")

# error = 200/np.array([0.21, 0.35, 0.72, 1.29, 4.1, 20.8, 158])
# yerr = np.array([0.01, 0.01, 0.01, 0.02, 0.16, 0.12, 0.47])
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Unknown correspondence (1024)")

# error = 200/np.array([0.15, 0.36, 0.66, 1.28, 2.48, 4.68, 8.9])
# yerr = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0])
# # plt.errorbar(np.arange(7), error, yerr=yerr, capsize=5)
# # plt.xticks(ticks=np.arange(7), labels=[4, 8, 16, 32, 64, 128, 256])
# plt.errorbar([4, 8, 16, 32, 64, 128, 256], error, yerr=yerr, capsize=5, label="Known correspondence (1024)")

# plt.xticks(ticks=[4, 16, 32, 64, 128, 256], rotation=45)
# ax.set_ylabel("Time (s)")
# plt.legend()
# ================================================


# perf circle
# ================================================
# unknown
# error = [0.5, 0.5, 0.46, 0.43, 0.4, 0.4, 0.39, 0.42, 0.45, 0.55, 1.32, 2.96, 7.3]
# yerr = [0.04, 0.02, 0.02, 0.00, 0.02, 0.01, 0, 0.02, 0, 0.01, 0.01, 0.01, 0.02]
# # plt.errorbar(np.arange(13), error, yerr=yerr, capsize=5)
# plt.errorbar([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], error, yerr=yerr, capsize=5, label="Unknown correspondence")

# # known
# error = [0.3, 0.3, 0.3, 0.3, 0.29, 0.3, 0.29, 0.32, 0.32, 0.38, 0.52, 0.98, 3.1]
# yerr = [0.03, 0.02, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
# # plt.errorbar(np.arange(13), error, yerr=yerr, capsize=5)
# plt.errorbar([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], error, yerr=yerr, capsize=5, label="Known correspondence")

# # PR
# error = [3.14, 6.17, 12.5, 23.26, 52.63, 90.9, 201.44, 333.3, 1021.51]
# yerr = [0.02, 0.02, 0.04, 0.07, 0.1, 0.12, 0.19, 0.26, 0.32]
# plt.errorbar([4, 8, 16, 32, 64, 128, 256, 512, 1024], error, yerr=yerr, capsize=5, label="Python Robotics (known)")

# # plt.xticks(ticks=np.arange(13), labels=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], rotation=45)
# plt.xticks(ticks=[4, 512, 1024, 2048, 4096, 8192, 16384], labels=[4, 512, 1024, 2048, 4096, 8192, 16384], rotation=45)
# ax.set_ylabel("Time (s)")
# ax.set_yscale("log")

# plt.legend()
# ================================================


# ax.set_ylabel("MSE(trans)")
# ax.plot([0,9], [0,0], "--", c="grey")
# plt.grid()

if EXPORT:
    plt.savefig('perf_circle.pgf')
else:
    plt.show()





#### PR error
# 4
# 1.5929832539609199 1.5169768767686533
# 0.7132846907548922 0.8048194465263544

# 8
# 1.0767257585708592 0.5964318188652005
# 0.5425414840435745 0.8342139114858463

# 16
# 1.4028758927419456 1.4892437933019245
# 0.5433993904389295 0.960542561114361

# 32
# 0.8226516339573049 0.597394658859978
# 0.26373720603890305 0.3784546352698212

# 64
# 0.9770508110859859 0.9187291630755422
# 0.41401784588532287 0.7658571341366375

# 128
# 0.880299372470477 0.9655017333542366
# 0.2667388653681361 0.6649521769159085
