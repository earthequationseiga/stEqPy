"""func15 on the full JMA data after one bkgd/prob pass: 1:1 port vs fused xint.
Prints wall time and the max differences in -logL and gradient.
Usage: python -I scripts/bench_xint.py data/jma/jap.in"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from steqpy import kernels as K  # noqa: E402
from steqpy.run import Model, readata  # noqa: E402

m = Model(readata(sys.argv[1]))
m.zbandw = K.bandwcalc(m.xx, m.yy, m.npp, m.bwm)
m.bkgdcalc()
m.probcalc()
quiet = lambda s: None  # noqa: E731
res = {}
for fast in (True, False, True, False):
    m.fast_xint = fast
    t0 = time.perf_counter()
    res[fast] = m.func15(m.x, quiet)
    print(f"fast_xint={fast!s:5s} func15 {time.perf_counter() - t0:8.2f} s  "
          f"f={res[fast][0]:.15f}", flush=True)
(f1, g1), (f0, g0) = res[True], res[False]
print(f"|df| {abs(f1 - f0):.3e}  max |dgrad| {np.max(np.abs(np.subtract(g1, g0))):.3e}")
