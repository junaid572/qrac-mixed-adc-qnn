"""Finite-shot accuracy of a trained detector (Section IV-B, Appendix B).

With mu(z) = (2c - 1) g(z) the signed logit and
    nu^2(z) = sum_i (dg/dz_i)^2 (1 - <Z_i>^2)                         (eq. (11))
the accuracy at s shots is approximately
    P_s(z) = Phi( sqrt(s) mu(z) / nu(z) ),                            (eq. (12))
and exactly the sum over all (s+1)^Q joint outcome counts of eq. (10).
"""
import itertools

import numpy as np
from scipy.stats import binom, norm

from .qnn import head


def margin_and_noise(z, y, params):
    """mu and nu^2 for every test symbol, from the exact readout z (N, Q)."""
    W1, b1, w2 = params["W1"], params["b1"], params["W2"][:, 0]
    active = (z @ W1 + b1) > 0                                     # ReLU units that are on
    grad = (active * w2[None, :]) @ W1.T                           # dg/dz, (N, Q)
    mu = (2.0 * y - 1.0) * head(z, params)
    nu2 = (grad ** 2 * (1.0 - z ** 2)).sum(axis=1)
    return mu, nu2


def closed_form_accuracy(z, y, params, shots):
    """Mean of eq. (12) over the test symbols, for each budget in `shots`."""
    mu, nu2 = margin_and_noise(z, y, params)
    live = nu2 > 1e-18                                             # nu = 0: no shot noise reaches the logit
    out = []
    for s in np.atleast_1d(shots):
        p = np.where(live, norm.cdf(np.sqrt(s) * mu / np.sqrt(np.where(live, nu2, 1.0))), mu > 0)
        out.append(float(p.mean()))
    return np.array(out)


def exact_accuracy(z, y, params, s, max_outcomes=20_000):
    """Eq. (10) by enumeration of every joint outcome (b_1..b_Q), b_i in {0..s}.
    Returns None when (s+1)^Q exceeds max_outcomes."""
    Q = z.shape[1]
    if (s + 1) ** Q > max_outcomes:
        return None
    grid = np.array(list(itertools.product(range(s + 1), repeat=Q)), dtype=float)   # (G, Q)
    correct = (head((2.0 * grid - s) / s, params) > 0)[None, :] == (y[:, None] > 0.5)  # (N, G)
    pmf = binom.pmf(np.arange(s + 1)[None, None, :], s, ((1.0 + z) / 2.0)[:, :, None])  # (N, Q, s+1)
    acc = 0.0
    for start in range(0, len(z), 100):                           # chunks keep memory small
        p = np.ones((min(100, len(z) - start), len(grid)))
        for i in range(Q):
            p *= pmf[start:start + 100, i, grid[:, i].astype(int)]
        acc += (p * correct[start:start + 100]).sum()
    return float(acc / len(z))
