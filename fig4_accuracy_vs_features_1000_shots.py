"""Fig. 4: test accuracy versus the number of features at 1000 shots (sigma = 2):
QRAC, the two angle encodings, the classical MLP, the perfect-CSI bound and the
CSI-free vote. Shaded bands: 0.5 std over channels of (accuracy - bound).

Uses the same simulations as Fig. 3 (shared cache in results/).

    python fig4_accuracy_vs_features_1000_shots.py --channels 5 --workers 8
"""
import matplotlib.pyplot as plt
import numpy as np

from qracsim import plotting, system
from qracsim.pipeline import run_all, simulate_detector

SIGMA, SHOTS = 2.0, 1000
K_ALL = list(range(10))
ARCHS = ["qrac", "angle_3q", "angle_bm"]


def jobs(channels):
    return [(a, k, c, SIGMA) for c in range(channels) for a in ARCHS + ["mlp"] for k in K_ALL]


def main():
    args = plotting.common_args(__doc__).parse_args()
    run_all(jobs(args.channels), args.workers)
    C = range(args.channels)
    n_feat = 2 + 2 * np.array(K_ALL)
    bound = np.array([[system.perfect_csi_accuracy(c, k, SIGMA) for c in C] for k in K_ALL])
    vote = np.array([[system.sign_vote_accuracy(c, k, SIGMA) for c in C] for k in K_ALL])
    mlp = np.array([[simulate_detector("mlp", k, c, SIGMA)["exact"] for c in C] for k in K_ALL])

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    plotting.line(ax, n_feat, mlp.mean(1), "mlp", marker=None)
    for arch in ARCHS:
        acc = np.array([[simulate_detector(arch, k, c, SIGMA)["shots"][SHOTS] for c in C] for k in K_ALL])
        plotting.line(ax, n_feat, acc.mean(1), arch, marker=None)
        plotting.band(ax, n_feat, acc.mean(1), acc, bound, arch)
    plotting.line(ax, n_feat, bound.mean(1), "bound")
    plotting.line(ax, n_feat, vote.mean(1), "vote")
    ax.set(xlabel="Number of features", ylabel="Test accuracy", ylim=(0.4, 1.0), xlim=(2, 20),
           title=f"{SHOTS} shots")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plotting.finish(fig, args.save)


if __name__ == "__main__":
    main()
