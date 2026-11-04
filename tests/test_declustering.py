"""The OOP API (StochasticDeclustering) must reproduce the validated driver (run.Model,
bit-identical to the Fortran over a full run) exactly: same params, probabilities, rates.
Schedule (2,) on 800 JMA events straddling the target-window start.
Run: python -I tests/test_declustering.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etas8py import Catalog, ETASParams, Region, Settings, StochasticDeclustering  # noqa
from etas8py import kernels as K  # noqa: E402
from etas8py.run import Model, readata  # noqa: E402

N = 800


def test_matches_driver():
    d = readata(ROOT / "data/jma/jap.in")
    i0 = int(np.searchsorted(d["zz"], d["tstart"])) - N // 2   # half before t_start: triggers
    for k in ("xx", "yy", "zz", "zmg", "ind"):
        d[k] = d[k][i0:i0 + N].copy()

    m = Model(d)                                 # reference: the 1:1 driver
    m.fast_xint = True
    m.zbandw = K.bandwcalc(m.xx, m.yy, m.npp, m.bwm)
    for _ in range(2):
        m.bkgdcalc()
        m.probcalc()
    with tempfile.TemporaryDirectory() as tmp:
        m.dav(Path(tmp), lambda s: None)

    sd = StochasticDeclustering(
        Catalog(d["xx"], d["yy"], d["zz"], d["zmg"]),
        Region(d["tx"], d["ty"], d["tstart"], d["tz"]),
        ETASParams.from_array(d["b"]),
        Settings(npp=d["npp"], bwm=d["bwm"], schedule=(2,), fast_xint=True),
    ).fit()

    assert np.array_equal(sd.ind, m.ind)
    assert np.array_equal(sd.params.to_array(), m.b)
    assert np.array_equal(sd.background_prob, m.zprob)
    assert np.array_equal(sd.background_rate, m.zbkgd)
    gx, gy = np.linspace(105, 120, 5), np.linspace(32, 40, 4)
    ref = K.outrates_grid(m.x, m.xx, m.yy, m.zz, m.zmg, m.zprob, m.zbandw, m.tz, m.tstart,
                          gx, gy)
    assert np.array_equal(sd.rates(gx, gy), ref)


if __name__ == "__main__":
    test_matches_driver()
    print("ok")
