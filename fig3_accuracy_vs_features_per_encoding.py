"""Fig. 3: test accuracy versus the number of features for each encoding, at
1, 10, 100, 1000 shots and under exact readout, with the perfect-CSI bound, the
CSI-free vote and the classical MLP (sigma = 2).

For every channel realization and every k = 0..9 active one-bit antennas, each
detector is trained on 3200 symbols and tested on 800 (qracsim/pipeline.py).
The basis encoding stops at k = 3, where it already needs 8 qubits.

    python fig3_accuracy_vs_features_per_encoding.py --channels 5 --workers 8
"""
import matplotlib.pyplot as plt
import numpy as np

from qracsim import plotting, system
from qracsim.pipeline import run_all, simulate_detector

SIGMA = 2.0
K_ALL = range(10)
K_RANGE = {"qrac": K_ALL, "elementary": range(4), "angle_3q": K_ALL, "angle_bm": K_ALL, "mlp": K_ALL}
SHOTS = [1, 10, 100, 1000]


def jobs(channels):
    return [(a, k, c, SIGMA) for c in range(channels) for a, ks in K_RANGE.items() for k in ks]


def main():
    args = plotting.common_args(__doc__).parse_args()
    run_all(jobs(args.channels), args.workers)                       # simulate (or load cached)
    C = range(args.channels)

    bound = np.array([[system.perfect_csi_accuracy(c, k, SIGMA) for c in C] for k in K_ALL])
    vote = np.array([[system.sign_vote_accuracy(c, k, SIGMA) for c in C] for k in K_ALL])
    mlp = np.array([[simulate_detector("mlp", k, c, SIGMA)["exact"] for c in C] for k in K_ALL])
    n_feat = 2 + 2 * np.array(K_ALL)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    ramp = plt.cm.Reds(np.linspace(0.35, 1.0, len(SHOTS) + 1))
    for ax, arch in zip(axes, ["qrac", "elementary", "angle_3q", "angle_bm"]):
        ks = list(K_RANGE[arch])
        res = [[simulate_detector(arch, k, c, SIGMA) for c in C] for k in ks]
        for s, col in zip(SHOTS + ["exact"], ramp):
            acc = [np.mean([r["exact"] if s == "exact" else r["shots"][s] for r in row]) for row in res]
            ax.plot(n_feat[ks], acc, color=col, lw=1.5, label="$\\infty$" if s == "exact" else f"{s} shots")
        ax.plot(n_feat, bound.mean(1), "k-", lw=1.2, label="Perfect CSI bound")
        ax.plot(n_feat, vote.mean(1), ":", color="gray", label="CSI-free vote")
        ax.plot(n_feat, mlp.mean(1), "k--", lw=1.2, label="MLP")
        ax.set(xlabel="Number of features", title=plotting.STYLE[arch][3], ylim=(0.4, 1.0),
               xlim=(2, n_feat[ks][-1] if len(ks) > 1 else 20))
    axes[0].set_ylabel("Test accuracy")
    axes[-1].legend(fontsize=7, loc="lower right")
    plotting.finish(fig, args.save)


if __name__ == "__main__":
    main()
