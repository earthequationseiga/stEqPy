"""Compare full-schedule outputs: Python port vs single-process Fortran vs shipped (10-rank MPI).

Shipped para/probs.dat/rates.dat are dated 2011 but jap.out 2016: they come from an older run
(final mu 0.2857 vs jap.out's 0.5506), so for "shipped" only jap.out (-logL, MLE) is compared.

Usage: python -I scripts/compare_runs.py
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "python": (ROOT / "results/python_full", "run.log"),
    "fastxint": (ROOT / "results/python_full_fastxint", "run.log"),
    "fortran": (ROOT / "build/fortran/full_run", "jap.out"),
    "fast_t10": (ROOT / "results/python_full_fast_t10", "run.log"),
    "mpi10": (ROOT / "build/fortran_mpi/full_run_np10", "jap.out"),
    "re_t10": (ROOT / "results/rerun_fast_t10", "run.log"),
    "re_mpi10": (ROOT / "build/fortran_mpi/rerun_np10", "jap.out"),
    "shipped": (ROOT / "original/sd_parallel", "jap.out"),
}


def load(d, logname):
    log = (d / logname).read_text()
    nll = [float(v.replace("D", "E")) for v in re.findall(r"- log likelihood =\s*(\S+)", log)]
    # final MLE from the log (Fortran wraps it over 2 lines), not from the para file
    para = np.array(log[log.rfind("mle =") + 5:].split()[:8], dtype=float)
    r = dict(nfunc=log.count("function value"), nll=nll[-1] if nll else float("nan"), para=para)
    if d.name != "sd_parallel":
        r["probs"] = np.loadtxt(d / "probs.dat")[:, 1:]
        r["rates"] = np.loadtxt(d / "rates.dat", skiprows=2)
    return r


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


if __name__ == "__main__":
    r = {k: load(*v) for k, v in RUNS.items()
         if (v[0] / "probs.dat").exists() or v[0].name == "sd_parallel"}  # skip unfinished runs
    for k, v in r.items():
        print(f"{k:8s} func15 calls {v['nfunc']:4d}  final -logL {v['nll']:.15f}")
        print(f"{'':8s} para " + " ".join(f"{p:.9f}" for p in v["para"]))
    pairs = (("python", "fortran"), ("python", "shipped"), ("fortran", "shipped"),
             ("fastxint", "python"), ("fastxint", "shipped"),
             ("fast_t10", "fastxint"), ("mpi10", "shipped"), ("mpi10", "fortran"),
             ("re_t10", "fast_t10"), ("re_mpi10", "mpi10"), ("re_mpi10", "fortran"))
    for a, b in pairs:
        if a not in r or b not in r:
            continue
        x, y = r[a], r[b]
        print(f"\n{a} vs {b}:")
        print(f"  -logL abs diff   {abs(x['nll'] - y['nll']):.3e}")
        print(f"  para max rel     {rel(x['para'], y['para']):.3e}")
        if "probs" not in y:
            continue
        for i, name in enumerate(("zprob", "zbandw", "zbkgd")):
            print(f"  {name:6s} max abs   {np.max(np.abs(x['probs'][:, i] - y['probs'][:, i])):.3e}")
        for i, name in enumerate(("bkgd", "kernel", "lambda")):
            print(f"  rates {name:6s} max rel {rel(x['rates'][:, i], y['rates'][:, i]):.3e}")
