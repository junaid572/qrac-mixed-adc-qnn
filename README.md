# QRAC-encoded QNN detection at a mixed-ADC receiver

Simulation code for *Qubit-Efficient Quantum Neural Networks for Mixed-ADC Receiver
Arrays* (IEEE Open Journal of the Communications Society). Each `figN_*.py` script
simulates the system and shows the corresponding figure of the paper in a
matplotlib window. No LaTeX installation is needed.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11 or later. Qiskit is needed only for Fig. 7.

## Run

```bash
python fig1_system_model.py
python fig2_encodings_bloch.py
python fig3_accuracy_vs_features_per_encoding.py --channels 5 --workers 8
python fig4_accuracy_vs_features_1000_shots.py   --channels 5 --workers 8
python fig5_accuracy_vs_shots.py                 --channels 5 --workers 8
python fig6_accuracy_vs_snr.py                   --channels 5 --workers 8
python fig7_noisy_device.py                      --channels 5 --workers 8
```

`--channels` is the number of Rayleigh channel realizations; the paper averages over
50 (`--channels 50`). `--workers` runs that many detectors in parallel, one CPU
thread each. `--save fig.pdf` also writes the figure to a file.

Every trained detector is cached in `results/`, so Figs. 3 to 5 and 7 reuse the same
simulations, and an interrupted run resumes where it stopped. Delete `results/` to
start from scratch. One detector takes about a minute on one core, so 50 channels of
Figs. 3 to 6 (about 5500 detectors) take a few hours on 20 cores.

## What is simulated

| step | where |
|---|---|
| Rayleigh channel `h_i ~ CN(0,1)`, BPSK, AWGN with `sigma = 2`; one high-resolution antenna (I and Q samples) and up to nine one-bit antennas (one comparator on I, one on Q) | `qracsim/system.py` |
| balanced 3200 / 800 training and test symbols per channel realization | `qracsim/system.py` |
| perfect-CSI log-likelihood detector (eq. (6)) and CSI-free sign vote | `qracsim/system.py` |
| basis, Angle (3Q), Angle (BM) and (3->1) QRAC encodings (eq. (8)) | `qracsim/encodings.py` |
| variational circuit (data layer, 4 layers of `R_X R_Y R_Z` + CNOT ring, final rotation), Pauli-Z readout, 10-unit ReLU head, Adam training with exact expectation values | `qracsim/qnn.py` |
| finite-shot readout, eq. (9): a budget of s shots is s circuit executions | `qracsim/qnn.py` |
| classical 16-8 MLP baseline | `qracsim/mlp.py` |
| finite-shot theory: exact enumeration (eq. (10)) and closed form (eq. (12)) | `qracsim/theory.py` |
| readout on the FakeTorino noise model of Qiskit | `qracsim/noisy.py` |
| one detector on one channel, end to end, with caching | `qracsim/pipeline.py` |

All seeds are fixed: channel realization `c` uses seed `c`, the data split and the
initialization use seed 0, and the shot outcomes of channel `c` with `k` one-bit
antennas use seed `97 + 131 c + k`. Results are reproducible on one machine; across
operating systems the trained detectors differ slightly through floating-point
rounding, which averages out over the channel realizations.
