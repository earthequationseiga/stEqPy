"""One call per expensive routine, same sequence as scripts/fortran/bench_main.f.

Numba compiles (or loads its cache) on a 200-event warm-up first, so timings exclude JIT.
Usage: python -I scripts/bench_python.py data/jma/jap.in
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from steqpy import kernels as K  # noqa: E402
from steqpy.run import Model, readata  # noqa: E402


def steps(m):
    m.zbandw = K.bandwcalc(m.xx, m.yy, m.npp, m.bwm)
    yield "bandwcalc"
    m.bkgdcalc()
    yield "bkgdcalc"
    m.probcalc()
    yield "probcalc"
    m.f, m.g = m.func15(m.x, log=lambda s: None)
    yield "func15"
    m.outrates(Path(m.outdir) / "rates.dat")
    yield "outrates"


def run(m):
    t0 = time.perf_counter()
    for name in steps(m):
        t1 = time.perf_counter()
        print(f" {name:>12s}{t1 - t0:12.3f} s", flush=True)
        t0 = t1


if __name__ == "__main__":
    out = Path("build/python_bench")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = readata(sys.argv[1])
    print(f" {'readata':>12s}{time.perf_counter() - t0:12.3f} s")

    small = dict(d)
    for k in ("xx", "yy", "zz", "zmg", "ind"):
        small[k] = d[k][:200]
    small["grid"] = (3, 3) + d["grid"][2:]
    w = Model(small)
    w.outdir = out
    t0 = time.perf_counter()
    for _ in steps(w):
        pass
    print(f" {'jit warm-up':>12s}{time.perf_counter() - t0:12.3f} s")

    m = Model(d)
    m.outdir = out
    run(m)
    print(" " + " ".join(f"{v:23.16E}" for v in [m.f] + list(m.g)))
    print(" " + " ".join(f"{v:23.16E}" for v in
                         [m.xint0, m.zbkgd[0], m.zbkgd[-1], m.zprob[1], m.zprob[-1]]))
