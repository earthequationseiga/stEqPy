"""Python davidn must reproduce the Fortran davidn (same eval count, same final x) on a
chained Rosenbrock. Needs build/fortran/test_davidn.exe (scripts/build_fortran.sh).
Run: python -I tests/test_davidn.py"""
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


if __name__ == "__main__":
    test_davidn_matches_fortran()
    print("ok")
