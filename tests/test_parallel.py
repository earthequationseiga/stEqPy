"""Parallel (prange) kernels must be bit-identical to the serial ones, at several thread
counts, on the first 2000 JMA events (keeps the test fast).
Run: python -I tests/test_parallel.py"""
import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etas8py import kernels as K  # noqa: E402
from etas8py.run import readata  # noqa: E402

N = 2000


def run(par, d):
    zprob = np.ones(N)
    zbkgd = np.zeros(N)
    bk = K.bkgdcalc_par if par else K.bkgdcalc
    pr = K.probcalc_par if par else K.probcalc
    fu = K.func15_par if par else K.func15
    xint0 = bk(d["xx"], d["yy"], zprob, d["zbandw"], d["tx"], d["ty"], d["npoly"], d["tz"],
               d["tstart"], zbkgd)
    b = np.sqrt(d["b"])
    pr(b, zbkgd, d["xx"], d["yy"], d["zz"], d["zmg"], zprob)
    grad = np.empty(8)
    f = fu(b, d["ind"], zbkgd, xint0, d["xx"], d["yy"], d["zz"], d["zmg"], d["tx"], d["ty"],
           d["npoly"], d["tz"], d["tstart"], grad, True)
    og = K.outrates_grid_par if par else K.outrates_grid
    gx = np.linspace(d["xx"].min(), d["xx"].max(), 15)
    gy = np.linspace(d["yy"].min(), d["yy"].max(), 12)
    rates = og(b, d["xx"], d["yy"], d["zz"], d["zmg"], zprob, d["zbandw"], d["tz"],
               d["tstart"], gx, gy)
    return np.r_[xint0, zbkgd, zprob, f, grad, rates.ravel()]


def test_parallel_bit_identical():
    d = readata(ROOT / "data/jma/jap.in")
    for k in ("xx", "yy", "zz", "zmg", "ind"):
        d[k] = d[k][:N].copy()
    d["zbandw"] = K.bandwcalc(d["xx"], d["yy"], d["npp"], d["bwm"])
    ref = run(False, d)
    for t in (1, 3, numba.config.NUMBA_NUM_THREADS):
        numba.set_num_threads(t)
        assert np.array_equal(run(True, d), ref), f"mismatch at {t} threads"


if __name__ == "__main__":
    test_parallel_bit_identical()
    print("ok")
