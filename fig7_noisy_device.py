"""Fig. 7: test accuracy versus the number of shots at k = 3 on the noisy emulation
of IBM's Torino device (solid) and with noiseless sampling (dashed).

The detectors are those of Fig. 5(a), trained without noise; only the readout is
moved to the FakeTorino noise model (qracsim/noisy.py). This step needs Qiskit and
takes a few minutes per channel.

    python fig7_noisy_device.py --channels 5 --workers 8
"""
import matplotlib.pyplot as plt
import numpy as np

from qracsim import plotting
from qracsim.pipeline import run_all, run_all_noisy, simulate_detector, simulate_noisy

SIGMA, K = 2.0, 3
SHOTS = [1, 10, 100, 1000]
X_INF = 1e4
ARCHS = ["qrac", "elementary", "angle_3q", "angle_bm"]


def jobs(channels):
    return [(a, K, c, SIGMA) for c in range(channels) for a in ARCHS + ["mlp"]]


def noisy_jobs(channels):
    return [(a, K, c) for c in range(channels) for a in ARCHS]


def main():
    args = plotting.common_args(__doc__).parse_args()
    run_all(jobs(args.channels), args.workers)                 # train (noiseless)
    run_all_noisy(noisy_jobs(args.channels), args.workers)     # read out on the noisy device
    C = range(args.channels)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.axhline(np.mean([simulate_detector("mlp", K, c, SIGMA)["exact"] for c in C]), color="k", ls="--", label="MLP")
    for arch in ARCHS:
        color, _, marker, label = plotting.STYLE[arch]
        clean = np.mean([[simulate_detector(arch, K, c, SIGMA)["shots"][s] for s in SHOTS] for c in C], 0)
        noisy = np.mean([[simulate_noisy(arch, K, c)[s] for s in SHOTS] for c in C], 0)
        exact = np.mean([simulate_detector(arch, K, c, SIGMA)["exact"] for c in C])
        ax.plot(SHOTS, noisy, color=color, marker=marker, ms=4, label=label)
        ax.plot(SHOTS + [X_INF], list(clean) + [exact], color=color, ls="--", lw=1)
    ax.plot([], [], "k-", label="noisy (Torino)")
    ax.plot([], [], "k--", lw=1, label="noiseless")
    ax.set_xscale("log")
    ax.set_xticks(SHOTS + [X_INF], ["$10^0$", "$10^1$", "$10^2$", "$10^3$", "$\\infty$"])
    ax.set(xlabel="Shots", ylabel="Test accuracy", ylim=(0.4, 1.0), title=f"k = {K}, FakeTorino")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    plotting.finish(fig, args.save)


if __name__ == "__main__":
    main()
