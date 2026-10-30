"""Shared matplotlib style (no LaTeX needed) and the bound-referenced bands of the figures."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})

STYLE = {  # color, line style, marker, label
    "mlp": ("black", "--", None, "MLP"),
    "qrac": ("tab:green", ":", "^", "QRAC"),
    "elementary": ("tab:orange", "-.", "D", "Elementary"),
    "angle_3q": ("tab:blue", "-", "o", "Angle encoding (3Q)"),
    "angle_bm": ("tab:red", "--", "s", "Angle encoding (BM)"),
    "bound": ("black", "-", None, "Perfect CSI bound"),
    "vote": ("gray", ":", None, "CSI-free vote"),
}


def line(ax, x, y, arch, **kw):
    c, ls, m, label = STYLE[arch]
    kw = {"color": c, "ls": ls, "marker": m, "ms": 4, "lw": 1.4, "label": label, **kw}
    return ax.plot(x, y, **kw)[0]


def band(ax, x, mean, per_channel, bound_per_channel, arch):
    """Shaded band of half-width 0.5 std over channels of (accuracy - perfect-CSI bound),
    which removes the channel-to-channel variation shared by all detectors."""
    hw = 0.5 * np.std(np.asarray(per_channel) - np.asarray(bound_per_channel), axis=-1)
    ax.fill_between(x, mean - hw, mean + hw, color=STYLE[arch][0], alpha=0.2, lw=0)


def common_args(description, default_channels=5):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--channels", type=int, default=default_channels,
                    help=f"channel realizations (paper: 50; default {default_channels})")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--save", default=None, help="also save the figure to this file")
    return ap


def finish(fig, save):
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    plt.show()
