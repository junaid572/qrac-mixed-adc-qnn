"""The simulation of one detector on one channel realization, end to end.

    simulate_detector(arch, k, channel_seed, sigma)

  1. draw the Rayleigh channel and pass 40 000 BPSK symbols through the mixed-ADC
     front end (system.simulate_front_end);
  2. keep the high-resolution antenna and k one-bit antennas, and split a balanced
     subset into 3200 training and 800 test symbols (system.train_test_split);
  3. for arch = "mlp": train the classical MLP and return its accuracy;
     otherwise encode the features into qubit rotations (encodings.ENCODERS[arch]),
  4. train the QNN with exact expectation values (qnn.QNNDetector.fit),
  5. read the test symbols out exactly (s = infinity) and with s = 1, 10, 100, 1000
     shots, ten independent readouts each (qnn.shot_accuracies).

Results are cached in results/ as small .npz files (one per detector), so that
figures sharing a simulation do not retrain, and an interrupted run resumes.
Delete results/ to simulate from scratch.
"""
import os
import time

import numpy as np

from . import encodings, qnn, system
from .mlp import mlp_accuracy

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _path(arch, k, channel_seed, sigma):
    return os.path.join(CACHE, f"sigma_{sigma:g}", f"channel_{channel_seed:02d}", f"{arch}_k{k}.npz")


def simulate_detector(arch, k, channel_seed, sigma=system.SIGMA, verbose=True):
    """Accuracies of one detector on one channel. Returns a dict with
    'exact' (s = infinity), 'shots' {s: accuracy}, and for the QNNs the trained
    'params' and the exact test readouts 'z' with labels 'y' (used by the
    finite-shot theory and the noisy-device emulation)."""
    path = _path(arch, k, channel_seed, sigma)
    if os.path.exists(path):
        return load(path)

    t0 = time.time()
    Xtr, ytr, Xte, yte = system.train_test_split(channel_seed, k, sigma)              # steps 1-2
    if arch == "mlp":
        res = {"exact": mlp_accuracy(Xtr, ytr, Xte, yte)}                             # step 3
    else:
        Atr, Ate = encodings.ENCODERS[arch](Xtr, Xte)                                 # step 3
        det = qnn.QNNDetector(Atr.shape[1], use_rz=(arch == "qrac")).fit(Atr, ytr)         # step 4
        z = np.asarray(det.expectations(Ate.astype(np.float32)))                      # step 5
        params = det.numpy_params()
        res = {"exact": det.exact_accuracy(Ate, yte),
               "shots": qnn.shot_accuracies(z, np.asarray(yte), params, channel_seed, k),
               "params": params, "z": z, "y": np.asarray(yte), "angles": Ate}
    save(path, res)
    if verbose:
        print(f"  sigma={sigma:g} channel {channel_seed:2d} k={k} {arch:10s} exact={res['exact']:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    return res


def save(path, res):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {"exact": res["exact"]}
    for s, a in res.get("shots", {}).items():
        flat[f"shot_{s}"] = a
    for name, v in res.get("params", {}).items():
        flat[f"param_{name}"] = v
    for key in ("z", "y", "angles"):
        if key in res:
            flat[key] = res[key]
    tmp = path + ".tmp.npz"
    np.savez(tmp, **flat)
    os.replace(tmp, path)


def load(path):
    d = np.load(path)
    res = {"exact": float(d["exact"])}
    shots = {int(n[5:]): float(d[n]) for n in d.files if n.startswith("shot_")}
    if shots:
        res["shots"] = shots
        res["params"] = {n[6:]: d[n] for n in d.files if n.startswith("param_")}
        res.update({key: d[key] for key in ("z", "y", "angles")})
    return res


# ---------------------------------------------------------------- running many
def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"


def run_all(jobs, workers=1):
    """Run simulate_detector over a list of (arch, k, channel_seed, sigma) jobs,
    optionally in parallel processes (one thread each). Cached jobs are skipped."""
    todo = [j for j in jobs if not os.path.exists(_path(*j))]
    print(f"{len(jobs)} detectors needed, {len(jobs) - len(todo)} cached, simulating {len(todo)}"
          f" with {workers} worker(s)", flush=True)
    _pool_map(_run_job, todo, workers)


def _run_job(job):
    simulate_detector(*job)


# ---------------------------------------------------------------- noisy device
def _noisy_path(arch, k, channel_seed):
    return os.path.join(CACHE, "faketorino", f"channel_{channel_seed:02d}", f"{arch}_k{k}.npz")


def simulate_noisy(arch, k, channel_seed, verbose=True):
    """Re-evaluate the trained detector (sigma = 2) on the FakeTorino noise model.
    Returns {s: accuracy} for s in 1, 10, 100, 1000."""
    path = _noisy_path(arch, k, channel_seed)
    if os.path.exists(path):
        d = np.load(path)
        return {int(n[5:]): float(d[n]) for n in d.files if n.startswith("shot_")}
    from . import noisy
    t0 = time.time()
    det = simulate_detector(arch, k, channel_seed, verbose=verbose)       # trained, noiseless
    out = noisy.noisy_shot_accuracies(noisy.make_simulator(), det["params"], det["angles"], det["y"],
                                      channel_seed, k, use_rz=(arch == "qrac"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, two_qubit_gates=out.pop("two_qubit_gates"), **{f"shot_{s}": v for s, v in out.items()})
    if verbose:
        print(f"  FakeTorino channel {channel_seed:2d} k={k} {arch:10s} 10 shots={out[10]:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    return out


def run_all_noisy(jobs, workers=1):
    """Like run_all, for (arch, k, channel_seed) noisy-device jobs."""
    todo = [j for j in jobs if not os.path.exists(_noisy_path(*j))]
    print(f"{len(jobs)} noisy evaluations needed, {len(jobs) - len(todo)} cached, running {len(todo)}", flush=True)
    _pool_map(_run_noisy, todo, workers)


def _run_noisy(job):
    simulate_noisy(*job)


def _pool_map(fn, todo, workers):
    if workers > 1 and len(todo) > 1:
        os.environ.update({"OMP_NUM_THREADS": "1",
                           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_worker_init) as pool:
            list(pool.map(fn, todo))
    else:
        for j in todo:
            fn(j)
