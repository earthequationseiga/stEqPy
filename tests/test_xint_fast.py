"""xint_fast (fused polygon integrals) must equal the 1:1 port xint to rounding, for every
output (value + 8 gradient terms), on a spread of JMA events and two parameter sets.
Run: python -I tests/test_xint_fast.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etas8py import kernels as K  # noqa: E402
from etas8py.run import readata  # noqa: E402

MLE = [0.550609, 0.165757, 0.029615, 1.657859, 1.153379, 0.001834, 1.950475, 1.067001]


def test_xint_fast_matches_reference():
    d = readata(ROOT / "data/jma/jap.in")
    args = (d["xx"], d["yy"], d["zz"], d["zmg"], d["tx"], d["ty"], d["npoly"], d["tz"],
            d["tstart"])
    worst = 0.0
    for theta in (d["b"], np.array(MLE)):
        b = np.sqrt(theta)
        for i in np.linspace(0, len(d["xx"]) - 1, 300).astype(int):
            h_ref, h_new = np.empty(8), np.empty(8)
            ref = np.r_[K.xint(i, b, *args, h_ref), h_ref]
            new = np.r_[K.xint_fast(i, b, *args, h_new), h_new]
            scale = np.maximum(np.abs(ref), 1e-300)
            worst = max(worst, float(np.max(np.abs(new - ref) / scale)))
    print(f"max rel diff {worst:.3e}")
    assert worst < 1e-10


if __name__ == "__main__":
    test_xint_fast_matches_reference()
    print("ok")
