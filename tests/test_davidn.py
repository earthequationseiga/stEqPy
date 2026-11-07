"""Python davidn must reproduce the Fortran davidn (same eval count, same final x) on a
chained Rosenbrock. Needs build/fortran/test_davidn.exe (scripts/build_fortran.sh).
Run: python -I tests/test_davidn.py"""
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy.davidn import davidn  # noqa: E402


def rosen(x, log):
    rosen.nev += 1
    n = len(x)
    f = 0.0
    g = [0.0] * n
    for i in range(n - 1):
        t1 = x[i + 1] - x[i] * x[i]
        t2 = 1.0 - x[i]
        f = f + 100.0 * t1 * t1 + t2 * t2
        g[i] = g[i] - 400.0 * x[i] * t1 - 2.0 * t2
        g[i + 1] = g[i + 1] + 200.0 * t1
    return f, g


def test_davidn_matches_fortran():
    exe = ROOT / "build" / "fortran" / "test_davidn.exe"
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stderr.split()
    f_nev, f_x = int(out[0]), [float(v) for v in out[1:9]]
    rosen.nev = 0
    x = davidn([-1.2 + 0.1 * i for i in range(1, 9)], rosen, lambda s: None)
    print("fortran", f_nev, f_x)
    print("python ", rosen.nev, x)
    assert rosen.nev == f_nev
    assert max(abs(a - b) for a, b in zip(x, f_x)) < 1e-12


def nan_below(x, log, bad=float("nan"), grad_only=False):
    """A bowl with its minimum at -1 and nothing usable below x[0] = 0, as with p < 1."""
    nan_below.calls += 1
    assert nan_below.calls < 2000, "search did not stop"
    f = sum((v + 1.0) ** 2 for v in x)
    g = [2.0 * (v + 1.0) for v in x]
    if x[0] < 0.0:
        if grad_only:
            return f, [bad] * len(x)
        return bad, [bad] * len(x)
    return f, g


def test_search_ends_on_values_that_are_not_finite():
    for name, kw in (("nan", {}), ("inf", {"bad": float("inf")}), ("-inf", {"bad": float("-inf")}),
                     ("gradient only", {"grad_only": True})):
        nan_below.calls = 0
        x = [1.0, 1.0, 1.0]
        davidn(x, lambda v, lg: nan_below(v, lg, **kw), lambda s: None)
        print(name, nan_below.calls, x)
        assert x[0] >= 0.0 and all(math.isfinite(v) for v in x), name
    # Already unusable at the start: returned untouched, no search.
    nan_below.calls = 0
    x = [-1.0, 1.0]
    davidn(x, nan_below, lambda s: None)
    assert x == [-1.0, 1.0] and nan_below.calls == 1


if __name__ == "__main__":
    test_davidn_matches_fortran()
    test_search_ends_on_values_that_are_not_finite()
    print("ok")
