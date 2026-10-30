"""Fig. 6: test accuracy versus the SNR of the active array at the matched-feature
cut (k = 3), at (a) 1000 and (b) 10 shots. The SNR is varied through the noise
level sigma; every detector is retrained at every sigma.

    python fig6_accuracy_vs_snr.py --channels 5 --workers 8
"""
import matplotlib.pyplot as plt
import numpy as np

from qracsim import plotting, system
from qracsim.pipeline import run_all, simulate_detector

K = 3
SIGMAS = [0.4472, 0.5, 0.6317, 0.7953, 1.0, 1.2604, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
ARCHS = ["qrac", "elementary", "angle_3q", "angle_bm"]


def jobs(channels):
    return [(a, K, c, s) for s in SIGMAS for c in range(channels) for a in ARCHS + ["mlp"]]


def main():
    args = plotting.common_args(__doc__).parse_args()
    run_all(jobs(args.channels), args.workers)
    C = range(args.channels)
    snr = system.active_array_snr_db(K, np.array(SIGMAS))
    bound = np.array([[system.perfect_csi_accuracy(c, K, s) for c in C] for s in SIGMAS])
    vote = np.array([[system.sign_vote_accuracy(c, K, s) for c in C] for s in SIGMAS])
    mlp = np.array([[simulate_detector("mlp", K, c, s)["exact"] for c in C] for s in SIGMAS])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, shots in zip(axes, (1000, 10)):
        for arch in ARCHS:
            acc = np.array([[simulate_detector(arch, K, c, s)["shots"][shots] for c in C] for s in SIGMAS])
            plotting.line(ax, snr, acc.mean(1), arch)
            plotting.band(ax, snr, acc.mean(1), acc, bound, arch)
        plotting.line(ax, snr, mlp.mean(1), "mlp")
        plotting.line(ax, snr, bound.mean(1), "bound")
        plotting.line(ax, snr, vote.mean(1), "vote")
        ax.set(xlabel="Array SNR (dB)", title=f"{shots} shots", ylim=(0.4, 1.0), xlim=(-10, 10))
    axes[0].set_ylabel("Test accuracy")
    axes[1].legend(fontsize=7, loc="upper left")
    plotting.finish(fig, args.save)


if __name__ == "__main__":
    main()
