"""Thread sweep for the prange kernels: one bkgdcalc + probcalc + func15 per thread count,
same sequence as the Fortran bench, then outrates (serial in the Fortran).
Usage: python -I scripts/bench_threads.py data/jma/jap.in [--slow-xint] 1 4 8 16
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numba  # noqa: E402

from etas8py import kernels as K  # noqa: E402
from etas8py.run import Model, readata  # noqa: E402

OUT = Path("build/python_bench")


def once(d, threads, fast):
    m = Model(d)
    m.fast_xint = fast
    m.parallel = threads > 0
    if threads > 0:
        numba.set_num_threads(threads)
        numba.set_parallel_chunksize(16)        # as in run.py
    m.zbandw = K.bandwcalc(m.xx, m.yy, m.npp, m.bwm)
    t = {}
    for name, fn in (("bkgdcalc", m.bkgdcalc), ("probcalc", m.probcalc),
                     ("func15", lambda: m.func15(m.x, log=lambda s: None)),
                     ("outrates", lambda: m.outrates(OUT / "rates.dat"))):
        t0 = time.perf_counter()
        r = fn()
        if name == "func15":
            f = r[0]
        t[name] = time.perf_counter() - t0
    return t, f


if __name__ == "__main__":
    args = sys.argv[2:]
    fast = "--slow-xint" not in args
    counts = [int(a) for a in args if a.isdigit()] or [0, 1, 4, 8, 16]
    OUT.mkdir(parents=True, exist_ok=True)
    d = readata(sys.argv[1])
    small = dict(d)
    for k in ("xx", "yy", "zz", "zmg", "ind"):
        small[k] = d[k][:200]
    small["grid"] = (3, 3) + d["grid"][2:]
    once(small, 0, fast)                        # JIT warm-up
    once(small, 1, fast)
    print(f"fast_xint={fast}  (threads 0 = serial loops)")
    for n in counts:
        t, f = once(d, n, fast)
        print(f" threads {n:2d}  " + "  ".join(f"{k} {v:8.3f} s" for k, v in t.items())
              + f"  f {f:.10f}", flush=True)
