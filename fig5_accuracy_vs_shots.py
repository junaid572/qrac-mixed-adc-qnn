"""Fig. 5: test accuracy versus the number of shots.

(a) matched-feature cut, k = 3 one-bit antennas (8 features): QRAC on 4 qubits,
    basis encoding on 8, Angle (3Q) on 3, Angle (BM) on 4. Thin lines are the closed
    form of eq. (12) and hollow markers the exact enumeration of eq. (10), both
    evaluated on the trained detectors (qracsim/theory.py).
(b) full array, k = 9 (20 features): QRAC and Angle (BM) on 8 qubits, Angle (3Q) on 3.

    python fig5_accuracy_vs_shots.py --channels 5 --workers 8
"""
import matplotlib.pyplot as plt
import numpy as np

from qracsim import plotting, system, theory
from qracsim.pipeline import run_all, simulate_detector

SIGMA = 2.0
SHOTS = [1, 10, 100, 1000]
X_INF = 1e4                                  # where "infinity" (exact readout) is drawn
PANELS = {3: ["qrac", "elementary", "angle_3q", "angle_bm"], 9: ["qrac", "angle_3q", "angle_bm"]}


def jobs(channels):
    return [(a, k, c, SIGMA) for c in range(channels) for k, archs in PANELS.items() for a in archs + ["mlp"]]


def main():
    args = plotting.common_args(__doc__).parse_args()
    run_all(jobs(args.channels), args.workers)
    C = range(args.channels)
    x = SHOTS + [X_INF]
    s_grid = np.logspace(0, 4, 60)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (k, archs) in zip(axes, PANELS.items()):
        bound = np.array([system.perfect_csi_accuracy(c, k, SIGMA) for c in C])
        ax.axhline(np.mean([simulate_detector("mlp", k, c, SIGMA)["exact"] for c in C]),
                   color="k", ls="--", lw=1.2, label="MLP")
        for arch in archs:
            res = [simulate_detector(arch, k, c, SIGMA) for c in C]
            acc = np.array([[r["shots"][s] for s in SHOTS] + [r["exact"]] for r in res]).T   # (5, channels)
            plotting.line(ax, x, acc.mean(1), arch)
            plotting.band(ax, x, acc.mean(1), acc, np.tile(bound, (len(x), 1)), arch)
            if k == 3:                                   # finite-shot theory on the trained detectors
                color = plotting.STYLE[arch][0]
                closed = np.mean([theory.closed_form_accuracy(r["z"], r["y"], r["params"], s_grid) for r in res], 0)
                ax.plot(s_grid, closed, color=color, lw=0.8)
                for s in (1, 10):
                    ex = [theory.exact_accuracy(r["z"], r["y"], r["params"], s) for r in res]
                    if None not in ex:
                        ax.plot(s, np.mean(ex), marker=plotting.STYLE[arch][2], ms=7, mfc="white",
                                mec=color, ls="none")
        ax.set_xscale("log")
        ax.set_xticks(x, ["$10^0$", "$10^1$", "$10^2$", "$10^3$", "$\\infty$"])
        ax.set(xlabel="Shots", ylim=(0.4, 1.0),
               title=f"({'a' if k == 3 else 'b'}) k = {k} ({2 + 2 * k} features)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Test accuracy")
    plotting.finish(fig, args.save)


if __name__ == "__main__":
    main()
