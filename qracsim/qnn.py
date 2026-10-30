"""Quantum neural network detector (Section IV-A) and the finite-shot readout model.

Circuit on Q qubits: data layer R_Y(theta_i) R_Z(phi_i) (R_Y alone for every encoding
except QRAC, whose azimuth phi_i is the only nonzero one), then L = 4 layers of trainable
R_X R_Y R_Z on every qubit followed by a CNOT ring, then a final trainable R_X R_Y R_Z
layer, and Pauli-Z readout z = (<Z_1>, ..., <Z_Q>). A classical head
g(z) = W2^T ReLU(W1^T z + b1) + b2 with 10 hidden units gives the logit, and the
decision is x_hat = sgn g(z). Training uses exact expectation values.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

N_LAYERS = 4
HIDDEN = 10
EPOCHS = 100
BATCH = 200
LR = 0.05
SEED = 0


def make_circuit(n_qubits, use_rz=True):
    """use_rz=False drops the R_Z data gates, whose angle is zero for every encoding but QRAC."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(angles, var_w, meas_w):
        for i in range(n_qubits):                                   # data layer
            qml.RY(angles[..., i, 0], wires=i)
            if use_rz:
                qml.RZ(angles[..., i, 1], wires=i)
        for layer in range(N_LAYERS):                               # variational layers
            for i in range(n_qubits):
                qml.RX(var_w[layer, i, 0], wires=i)
                qml.RY(var_w[layer, i, 1], wires=i)
                qml.RZ(var_w[layer, i, 2], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):                                   # final rotation
            qml.RX(meas_w[i, 0], wires=i)
            qml.RY(meas_w[i, 1], wires=i)
            qml.RZ(meas_w[i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit


def init_params(n_qubits, seed=SEED):
    """Circuit angles uniform in [-0.05, 0.05], head Glorot-uniform, zero biases."""
    k1, k2, k3, k4, _ = jax.random.split(jax.random.PRNGKey(seed), 5)
    glorot = lambda k, shape: jax.random.uniform(k, shape, minval=-jnp.sqrt(6.0 / sum(shape)),
                                                 maxval=jnp.sqrt(6.0 / sum(shape)))
    return {"var_weights": jax.random.uniform(k1, (N_LAYERS, n_qubits, 3), minval=-0.05, maxval=0.05),
            "meas_weights": jax.random.uniform(k2, (n_qubits, 3), minval=-0.05, maxval=0.05),
            "W1": glorot(k3, (n_qubits, HIDDEN)), "b1": jnp.zeros((HIDDEN,)),
            "W2": glorot(k4, (HIDDEN, 1)), "b2": jnp.zeros((1,))}


class QNNDetector:
    def __init__(self, n_qubits, use_rz=True, seed=SEED):
        self.n_qubits, self.use_rz = n_qubits, use_rz
        self.circuit = make_circuit(n_qubits, use_rz)
        self.params = init_params(n_qubits, seed)

    def expectations(self, angles, params=None):
        """Exact readout <Z_i> for every symbol, shape (N, Q)."""
        p = self.params if params is None else params
        return jnp.stack(self.circuit(angles, p["var_weights"], p["meas_weights"]), axis=-1)

    def logits(self, params, angles):
        z = self.expectations(angles, params)
        return (jax.nn.relu(z @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]).squeeze(-1)

    def fit(self, angles, labels, seed=SEED):
        """Adam on the binary cross-entropy, mini-batches of 200, 100 epochs."""
        opt = optax.adam(LR)
        state = opt.init(self.params)
        loss = lambda p, a, y: jnp.mean(optax.sigmoid_binary_cross_entropy(self.logits(p, a), y))

        @jax.jit
        def step(p, s, a, y):
            g = jax.grad(loss)(p, a, y)
            upd, s = opt.update(g, s, p)
            return optax.apply_updates(p, upd), s

        A = jnp.asarray(angles, dtype=jnp.float32)
        Y = jnp.asarray(labels, dtype=jnp.float32)
        rng, params = np.random.default_rng(seed), self.params
        for _ in range(EPOCHS):
            perm = rng.permutation(len(A))
            for start in range(0, len(A), BATCH):
                idx = perm[start:start + BATCH]
                params, state = step(params, state, A[idx], Y[idx])
        self.params = params
        return self

    def exact_accuracy(self, angles, labels):
        """Accuracy with exact expectation values (s = infinity)."""
        lg = jax.jit(self.logits)(self.params, jnp.asarray(angles, dtype=jnp.float32))
        return float(jnp.mean(((lg > 0.0).astype(jnp.float32) == jnp.asarray(labels)).astype(jnp.float32)))

    def numpy_params(self):
        return {k: np.asarray(v) for k, v in self.params.items()}


# ---------------------------------------------------------------- finite shots
SHOTS = [1, 10, 100, 1000]
N_REPEATS = 10


def head(z, params):
    """Classical head g(z) evaluated in numpy on (estimated) readouts."""
    h = np.maximum(z @ params["W1"] + params["b1"], 0.0)
    return (h @ params["W2"] + params["b2"]).squeeze(-1)


def shot_readout(z, shots, rng):
    """Eq. (9): each <Z_i> is replaced by the mean of `shots` outcomes in {-1,+1}
    with Pr[+1] = (1 + <Z_i>) / 2. A budget of s shots is s circuit executions."""
    p = (1.0 + z) / 2.0
    return ((rng.random((shots,) + z.shape) < p[None]) * 2.0 - 1.0).mean(axis=0)


def shot_accuracies(z, labels, params, channel_seed, k):
    """Mean test accuracy over N_REPEATS independent readouts, for every budget in SHOTS."""
    rng = np.random.default_rng(97 + 131 * channel_seed + k)
    out = {}
    for s in SHOTS:
        accs = [((head(shot_readout(z, s, rng), params) > 0.0) == labels).mean() for _ in range(N_REPEATS)]
        out[s] = float(np.mean(accs))
    return out
