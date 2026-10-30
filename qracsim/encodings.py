"""Quantum feature encodings (Section III).

Every encoder maps the feature matrix [Re y_0, Im y_0, bits...] of the training and
test symbols to rotation angles of shape (N, Q, 2): one (theta, phi) pair per qubit,
prepared from |0> as R_Y(theta) followed by R_Z(phi). The two continuous features are
min-max scaled on the training split and loaded as R_Y(pi v) on two qubits in all
encodings; the encodings differ only in how the 2k bits are placed.
"""
import numpy as np


def _minmax(train_cont, cont):
    lo, hi = train_cont.min(axis=0), train_cont.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((cont - lo) / span, 0.0, 1.0)


def _ry(values):
    """R_Y(pi v) with phi = 0, for bits (v in {0,1}) and scaled continuous values."""
    theta = np.asarray(values, dtype=np.float64) * np.pi
    return np.stack([theta, np.zeros_like(theta)], axis=-1)


def qrac_3to1(bits):
    """(3->1) QRAC, eq. (8): bit triple b1 b2 b3 -> Bloch vector (2b-1)/sqrt(3)."""
    n = bits.shape[0]
    v = (2.0 * bits.reshape(n, -1, 3) - 1.0) / np.sqrt(3.0)
    theta = np.arccos(np.clip(v[..., 2], -1.0, 1.0))
    phi = np.arctan2(v[..., 1], v[..., 0])
    return np.stack([theta, phi], axis=-1)


def qrac_2to1(bits):
    """(2->1) QRAC for a leftover pair: equatorial Bloch vector (2b1-1, 2b2-1, 0)/sqrt(2)."""
    n = bits.shape[0]
    v = 2.0 * bits.reshape(n, -1, 2) - 1.0
    theta = np.full(v.shape[:2], np.pi / 2.0)
    return np.stack([theta, np.arctan2(v[..., 1], v[..., 0])], axis=-1)


def integer_angle(bits):
    """Angle encoding of a bit group read as an integer t (MSB first): R_Y(pi t / (2^m - 1))."""
    m = bits.shape[1]
    t = bits.astype(np.float64) @ (2.0 ** np.arange(m - 1, -1, -1))
    theta = t / (2.0 ** m - 1.0) * np.pi
    return np.stack([theta, np.zeros_like(theta)], axis=-1)


def _split(Xtr, Xte):
    return Xtr[:, :2], Xtr[:, 2:], Xte[:, :2], Xte[:, 2:]


def elementary(Xtr, Xte):
    """Basis encoding: one qubit per feature (bits first, then the two continuous)."""
    ctr, btr, cte, bte = _split(Xtr, Xte)
    enc = lambda b, c: _ry(np.concatenate([b, _minmax(ctr, c)], axis=1))
    return enc(btr, ctr), enc(bte, cte)


def qrac(Xtr, Xte):
    """QRAC encoding: bits packed three per qubit; a leftover of one bit gets an R_Y
    qubit and a leftover of two bits a (2->1) qubit. Then the two continuous qubits.
    Q = ceil(2k/3) + 2."""
    ctr, btr, cte, bte = _split(Xtr, Xte)

    def enc(b, c):
        parts, rem = [], b.shape[1] % 3
        if rem == 1:
            parts.append(_ry(b[:, :1]))
        elif rem == 2:
            parts.append(qrac_2to1(b[:, :2]))
        if b.shape[1] > rem:
            parts.append(qrac_3to1(b[:, rem:]))
        return np.concatenate(parts + [_ry(_minmax(ctr, c))], axis=1)
    return enc(btr, ctr), enc(bte, cte)


def angle_3q(Xtr, Xte):
    """Angle encoding (3Q): all 2k bits as one integer on one qubit, Q = 3."""
    ctr, btr, cte, bte = _split(Xtr, Xte)

    def enc(b, c):
        parts = [integer_angle(b)[:, None, :]] if b.shape[1] else []
        return np.concatenate(parts + [_ry(_minmax(ctr, c))], axis=1)
    return enc(btr, ctr), enc(bte, cte)


def angle_bm(Xtr, Xte):
    """Angle encoding (BM): groups of up to three bits as integers, Q = ceil(2k/3) + 2,
    the same budget as the QRAC encoding."""
    ctr, btr, cte, bte = _split(Xtr, Xte)

    def enc(b, c):
        parts = [integer_angle(b[:, g:g + 3])[:, None, :] for g in range(0, b.shape[1], 3)]
        return np.concatenate(parts + [_ry(_minmax(ctr, c))], axis=1)
    return enc(btr, ctr), enc(bte, cte)


ENCODERS = {"qrac": qrac, "elementary": elementary, "angle_3q": angle_3q, "angle_bm": angle_bm}


def bloch_vector(theta, phi):
    """Bloch vector of R_Z(phi) R_Y(theta)|0>."""
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
