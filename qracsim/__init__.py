"""Helpers for simulating QRAC-encoded QNN detectors at a mixed-ADC receiver.

system     channel, mixed-ADC front end, data split, perfect-CSI and sign-vote references
encodings  basis, Angle (3Q), Angle (BM) and QRAC encodings into qubit rotations
qnn        variational circuit, training, exact and finite-shot readout
mlp        classical MLP baseline
pipeline   simulate_detector(): one detector on one channel, end to end (cached)
theory     finite-shot accuracy: exact enumeration and closed form (Section IV-B)
noisy      emulation on the FakeTorino noise model (Section V-C)
plotting   matplotlib style without LaTeX, bound-referenced bands
"""
