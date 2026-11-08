"""pyperf benchmarks of the per-iteration kernels on JMA (10071 events, 4656-4658 targets).

One call each of: -log L + gradient (K form = default, A form with s-GL, preset "original"
= Fortran's ndiv path), background update, probability update. These dominate a fit (a fit is
roughly calls x likelihood + outers x (background + probabilities)), so a change in a kernel
shows here with mean +- std, without the minutes and noise of a whole fit. Values do not depend
on the parameters much (same pairs, same power laws), so the initial parameters are used.

pyperf runs each benchmark in fresh worker processes and warns when the spread is large; this PC
can run 3-6x slow without high CPU load, so check the warnings. Whole fits: scripts/timing_suite.py.

    python -I scripts/bench_kernels.py -o build/bench/kernels.json
    python -I scripts/bench_kernels.py --fast            # fewer processes, quick look
    python -m pyperf compare_to build/bench/old.json build/bench/kernels.json
"""
import sys
from pathlib import Path

import numpy as np
import pyperf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy import StochasticDeclustering  # noqa: E402

THREADS = 10


_models = {}


def model(kind):
    """Built on first use: each pyperf worker runs one benchmark, so it builds only the model
    that benchmark needs, inside the warmup call (whose time pyperf discards)."""
    if kind not in _models:
        over = {"K": {}, "A": {"form": "A"}, "original": {"preset": "original"}}[kind]
        sd = StochasticDeclustering.from_config(ROOT / "configs/jma.toml", threads=THREADS, **over)
        sd.compute_bandwidths()
        sd.update_background()
        _models[kind] = sd
    return _models[kind]


def loglik(kind):
    sd = model(kind)
    sd._func15(np.sqrt(sd._theta()), sd.xint0)


def main():
    # 1 warmup (builds the model, loads Numba's disk cache), then 3 values per process
    runner = pyperf.Runner(processes=6, values=3, warmups=1, loops=1)
    runner.metadata["threads"] = THREADS
    runner.metadata["catalogue"] = "JMA M>=4.5 (configs/jma.toml)"
    runner.bench_func("loglik_grad K (default)", loglik, "K")
    runner.bench_func("loglik_grad A sgl", loglik, "A")
    runner.bench_func("loglik_grad original ndiv", loglik, "original")
    runner.bench_func("background sgl", lambda: model("K").update_background())
    runner.bench_func("background original ndiv", lambda: model("original").update_background())
    runner.bench_func("probabilities K (default)", lambda: model("K").update_probabilities())
    runner.bench_func("probabilities original", lambda: model("original").update_probabilities())


if __name__ == "__main__":
    main()
