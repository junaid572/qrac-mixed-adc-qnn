"""Emulation of a trained detector on a noisy device (Section V-C).

The trained circuit is rebuilt in Qiskit with its weights as constants, transpiled
to the native gates of IBM's FakeTorino calibration snapshot with routing restricted
to the physical chain 0..Q-1, and sampled on the density-matrix simulator of
Qiskit Aer with the snapshot's gate, relaxation and readout errors. Each shot measures
all Q qubits, and the joint outcomes are averaged per qubit before the classical head.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from .qnn import N_REPEATS, SHOTS, head


def make_simulator():
    return AerSimulator.from_backend(FakeTorino(), method="density_matrix", max_parallel_experiments=0)


def build_circuit(params, n_qubits, use_rz):
    theta = [Parameter(f"th{i}") for i in range(n_qubits)]
    phi = [Parameter(f"ph{i}") for i in range(n_qubits)] if use_rz else []
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(theta[i], i)
        if use_rz:
            qc.rz(phi[i], i)
    for layer in params["var_weights"]:
        for i in range(n_qubits):
            qc.rx(float(layer[i, 0]), i)
            qc.ry(float(layer[i, 1]), i)
            qc.rz(float(layer[i, 2]), i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    for i in range(n_qubits):
        qc.rx(float(params["meas_weights"][i, 0]), i)
        qc.ry(float(params["meas_weights"][i, 1]), i)
        qc.rz(float(params["meas_weights"][i, 2]), i)
    qc.measure_all()
    return qc, theta + phi


def noisy_shot_accuracies(sim, params, angles, labels, channel_seed, k, use_rz=True):
    """Accuracy at every budget in SHOTS on the emulated device (N_REPEATS readouts each)."""
    n, Q = angles.shape[0], angles.shape[1]
    qc, data_params = build_circuit(params, Q, use_rz)
    chain = CouplingMap([(i, i + 1) for i in range(Q - 1)] + [(i + 1, i) for i in range(Q - 1)])
    tqc = transpile(qc, sim, coupling_map=chain, initial_layout=list(range(Q)),
                    optimization_level=1, seed_transpiler=0)
    total = N_REPEATS * sum(SHOTS)              # one long outcome stream per symbol, sliced below
    zhat = {(s, r): [] for s in SHOTS for r in range(N_REPEATS)}
    seed = 97 + 131 * channel_seed + k
    for lo in range(0, n, 100):
        rows = range(lo, min(lo + 100, n))
        values = lambda i: [float(a) for a in angles[i, :, 0]] + ([float(a) for a in angles[i, :, 1]] if use_rz else [])
        bound = [tqc.assign_parameters(dict(zip(data_params, values(i)))) for i in rows]
        result = sim.run(bound, shots=total, memory=True, seed_simulator=seed + lo).result()
        for j in range(len(rows)):
            mem = result.get_memory(j)
            bits = np.frombuffer("".join(m[::-1] for m in mem).encode(), dtype=np.uint8).reshape(total, Q) - ord("0")
            pm, ptr = 1.0 - 2.0 * bits, 0
            for r in range(N_REPEATS):
                for s in SHOTS:
                    zhat[s, r].append(pm[ptr:ptr + s].mean(axis=0))
                    ptr += s
    out = {}
    for s in SHOTS:
        out[s] = float(np.mean([((head(np.stack(zhat[s, r]), params) > 0) == labels).mean()
                                for r in range(N_REPEATS)]))
    out["two_qubit_gates"] = sum(1 for inst in tqc.data if inst.operation.num_qubits == 2)
    return out
