"""System model: Rayleigh channel, mixed-ADC front end, data split, and the two
classical references (perfect-CSI log-likelihood detector and CSI-free sign vote).

    y_i = h_i x + n_i,   h_i ~ CN(0,1),   n_i ~ CN(0, 2 sigma^2),   x in {-1,+1}

Antenna 0 is high resolution and delivers (Re y_0, Im y_0). Antennas 1..9 are
one-bit: each delivers the two comparator outputs (1{Re y_i > 0}, 1{Im y_i > 0}).
With k active one-bit antennas the detector sees n = 2 + 2k features.
"""
from functools import lru_cache

import numpy as np
from scipy.stats import norm

M_ANT = 10            # antennas: 1 high-resolution + 9 one-bit
M_C = 1               # high-resolution antennas
MAX_MB = 9            # one-bit antennas
SIGMA = 2.0           # noise std per real dimension (array SNR 0.97 dB at k = 9)
N_POOL = 40_000       # symbols simulated per channel realization
N_PER_CLASS = 2_000   # balanced subset: 2000 symbols per class
TRAIN_FRAC = 0.8      # 3200 training / 800 test symbols
SPLIT_SEED = 0        # seed of the subset and of the split


def draw_channel(rng):
    """h_i ~ CN(0,1), i = 0..M_ANT-1."""
    return (rng.normal(0.0, 1.0, M_ANT) + 1j * rng.normal(0.0, 1.0, M_ANT)) / np.sqrt(2.0)


@lru_cache(maxsize=8)
def simulate_front_end(channel_seed, sigma=SIGMA):
    """One block-fading realization: N_POOL symbols through the mixed-ADC front end.

    Returns (h, cont, bits, label) with
      cont  (N, 2)  : Re y_0, Im y_0 of the high-resolution antenna
      bits  (N, 18) : I comparators of antennas 1..9, then Q comparators of 1..9
      label (N,)    : 1{x = +1}
    """
    rng = np.random.default_rng(channel_seed)
    h = draw_channel(rng)                                         # fixed within the block
    x = rng.choice([-1.0, 1.0], size=N_POOL)                      # BPSK symbols
    noise = rng.normal(0.0, sigma, (N_POOL, M_ANT)) + 1j * rng.normal(0.0, sigma, (N_POOL, M_ANT))
    y = h[None, :] * x[:, None] + noise
    cont = np.concatenate([y[:, :M_C].real, y[:, :M_C].imag], axis=1)
    bits = np.concatenate([(y[:, M_C:].real > 0), (y[:, M_C:].imag > 0)], axis=1).astype(np.float64)
    return h, cont, bits, (x > 0).astype(np.int64)


def features(channel_seed, k, sigma=SIGMA):
    """Feature matrix with k active one-bit antennas: [Re y_0, Im y_0, I-bits(1..k), Q-bits(1..k)]."""
    _, cont, bits, label = simulate_front_end(channel_seed, sigma)
    cols = np.r_[0:k, MAX_MB:MAX_MB + k]
    return np.concatenate([cont, bits[:, cols]], axis=1), label


def train_test_split(channel_seed, k, sigma=SIGMA):
    """Balanced 4000-symbol subset of the block, split 3200/800 per class.

    Returns Xtr, ytr, Xte, yte (labels as float32). The subset does not depend on k,
    so every encoding and every k is trained and tested on the same symbols."""
    X, y = features(channel_seed, k, sigma)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == 1), N_PER_CLASS, replace=False),
                          rng.choice(np.flatnonzero(y == 0), N_PER_CLASS, replace=False)])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    rng = np.random.default_rng(SPLIT_SEED)
    tr, te = [], []
    for c in (0, 1):
        ci = np.flatnonzero(y == c)
        rng.shuffle(ci)
        n_tr = int(round(TRAIN_FRAC * len(ci)))
        tr.append(ci[:n_tr])
        te.append(ci[n_tr:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    rng.shuffle(tr)
    rng.shuffle(te)
    return X[tr], y[tr].astype(np.float32), X[te], y[te].astype(np.float32)


# ---------------------------------------------------------------- references
N_REF = 200_000       # symbols used to evaluate the two references


def _reference_symbols(channel_seed, k, sigma):
    """Same channel as the detectors, fresh symbols for evaluating the references."""
    h = draw_channel(np.random.default_rng(channel_seed))[:M_C + k]
    rng = np.random.default_rng(10_000 + channel_seed)
    x = rng.choice([-1.0, 1.0], size=N_REF)
    noise = rng.normal(0.0, sigma, (N_REF, len(h))) + 1j * rng.normal(0.0, sigma, (N_REF, len(h)))
    return h, x, h[None, :] * x[:, None] + noise


def perfect_csi_accuracy(channel_seed, k, sigma=SIGMA):
    """Accuracy of the maximum-likelihood detector that knows h and sigma (eq. (6)):
    matched filter on the high-resolution antenna plus a signed log-reliability per bit."""
    h, x, y = _reference_symbols(channel_seed, k, sigma)
    llr = (2.0 * (h[0].real * y[:, 0].real + h[0].imag * y[:, 0].imag) / sigma ** 2)
    if k:
        b = np.concatenate([y[:, M_C:].real > 0, y[:, M_C:].imag > 0], axis=1).astype(float)
        g = np.concatenate([h[M_C:].real, h[M_C:].imag])          # gain seen by each comparator
        w = np.log(np.clip(norm.cdf(g / sigma), 1e-300, 1) / np.clip(norm.cdf(-g / sigma), 1e-300, 1))
        llr = llr + ((2.0 * b - 1.0) * w[None, :]).sum(1)
    return float((np.sign(llr) == x).mean())


def sign_vote_accuracy(channel_seed, k, sigma=SIGMA):
    """CSI-free unweighted vote over the one-bit outputs: chance level, since the
    signs of Re h_i and Im h_i are random."""
    if k == 0:
        return 0.5
    _, x, y = _reference_symbols(channel_seed, k, sigma)
    b = np.concatenate([y[:, M_C:].real > 0, y[:, M_C:].imag > 0], axis=1).astype(float)
    return float((np.sign((2.0 * b - 1.0).sum(1) + 1e-12) == x).mean())


def active_array_snr_db(k, sigma):
    """SNR of the active array, (1 + k) / (2 sigma^2), in dB."""
    return 10.0 * np.log10((M_C + k) / (2.0 * sigma ** 2))
