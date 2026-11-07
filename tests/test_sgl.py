"""s-GL polygon integrals against answers that do not use them:
- ETAS kernel with q = 2 has a closed form per edge; its finite differences check the D and
  gamma integrands, on easy and awkward centres (on an edge, 1e-6 from it, on its line).
- Gaussian: a polygon much larger than the kernel holds all of it; else equal-length pieces.
- func15_sgl's gradient matches finite differences of its value; serial == parallel.
- Preset "original" stays on the equal-length pieces.
Run: python -I tests/test_sgl.py"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy import Catalog, ETASParams, Region, Settings, StochasticDeclustering  # noqa: E402
from steqpy import config  # noqa: E402
from steqpy import kernels as K  # noqa: E402
from steqpy.run import readata  # noqa: E402

POLYGONS = {"triangle": ([0.0, 10.0, 3.0, 0.0], [0.0, 0.0, 8.0, 0.0]),
            "needle": ([0.0, 10.0, 5.0, 0.0], [0.0, 0.0, 0.001, 0.0]),
            "non-convex": ([0.0, 6.0, 6.0, 3.0, 0.0, 0.0], [0.0, 0.0, 5.0, 1.0, 5.0, 0.0])}


def exact_q2(ssig, xp, yp, xc, yc):
    """q = 2: F = r^2 / (r^2 + ssig) / 2 pi, so int F dphi = h / R [atan(t / R)] / 2 pi over
    an edge, with h the signed distance to its line, t the position along it, R^2 = h^2 + ssig."""
    s = 0.0
    for j in range(len(xp) - 1):
        dx, dy = xp[j + 1] - xp[j], yp[j + 1] - yp[j]
        L = math.hypot(dx, dy)
        ax, ay, bx, by = xp[j] - xc, yp[j] - yc, xp[j + 1] - xc, yp[j + 1] - yc
        h = (ax * dy - ay * dx) / L
        R = math.sqrt(h * h + ssig)
        s += h / R * (math.atan((bx * dx + by * dy) / L / R) - math.atan((ax * dx + ay * dy) / L / R))
    return s / 2 / math.pi


def centres(xp, yp):
    mx, my = 0.5 * (xp[0] + xp[1]), 0.5 * (yp[0] + yp[1])
    return [(np.mean(xp[:-1]), np.mean(yp[:-1])), (xp[0], yp[0]), (mx, my), (mx, my + 1e-6),
            (mx, my - 1e-6), (2 * xp[1] - xp[0], 2 * yp[1] - yp[0]), (100.0, 37.0)]


def test_etas_kernel_and_integrands_against_closed_form():
    gx, gw = K.gauss_nodes(128)
    e = 1e-5
    for name, (x, y) in POLYGONS.items():
        xp, yp = np.array(x), np.array(y)
        for xc, yc in centres(xp, yp):
            for d, gamma, m in ((1e-4, 1.0, 0.0), (0.01, 1.2, 1.5), (1.0, 0.8, 3.0)):
                ssig = d * math.exp(gamma * m)
                s0, s1, _, s3 = K.polyint_sgl4(gamma, d, 2.0, m, xp, yp, len(xp) - 1, xc, yc, gx, gw)
                where = (name, xc, yc, d, m)
                assert abs(s0 - exact_q2(ssig, xp, yp, xc, yc)) < 1e-11, where
                fd = (exact_q2(d * (1 + e) * math.exp(gamma * m), xp, yp, xc, yc)
                      - exact_q2(d * (1 - e) * math.exp(gamma * m), xp, yp, xc, yc)) / (2 * e * d)
                assert abs(s1 - fd) < 1e-6 * max(1.0, abs(fd)), where
                if m > 0:
                    fd = (exact_q2(d * math.exp(gamma * (1 + e) * m), xp, yp, xc, yc)
                          - exact_q2(d * math.exp(gamma * (1 - e) * m), xp, yp, xc, yc)) / (2 * e * gamma)
                    assert abs(s3 - fd) < 1e-6 * max(1.0, abs(fd)), where


def test_q_integrand_against_finite_difference():
    gx, gw = K.gauss_nodes(128)
    xp, yp = map(np.array, POLYGONS["triangle"])
    for xc, yc in centres(xp, yp):
        for q in (1.5, 1.95, 3.0):
            args = (xp, yp, 3, xc, yc, gx, gw)
            e = 1e-6
            fd = (K.polyint_sgl4(1.0, 0.01, q + e, 1.0, *args)[0]
                  - K.polyint_sgl4(1.0, 0.01, q - e, 1.0, *args)[0]) / (2 * e)
            assert abs(K.polyint_sgl4(1.0, 0.01, q, 1.0, *args)[2] - fd) < 1e-7, (xc, yc, q)


def test_gaussian():
    gx, gw = K.gauss_nodes(64)
    side = 100.0
    xp, yp = np.array([0.0, side, side, 0.0, 0.0]), np.array([0.0, 0.0, side, side, 0.0])
    # bandwidth 1 at the centre of a 100 x 100 square: the rest is 1e-500 away from mattering
    assert abs(K.polyint_gauss_sgl(1.0, xp, yp, 4, 50.0, 50.0, gx, gw) - 1.0) < 1e-12
    # half of the kernel lies inside when the centre is on a long straight edge
    assert abs(K.polyint_gauss_sgl(1.0, xp, yp, 4, 50.0, 0.0, gx, gw) - 0.5) < 1e-12
    # outside the polygon, nothing
    assert abs(K.polyint_gauss_sgl(0.1, xp, yp, 4, 500.0, 500.0, gx, gw)) < 1e-12
    xp, yp = map(np.array, POLYGONS["triangle"])
    for xc, yc in ((4.0, 3.0), (0.5, 0.5), (20.0, 3.0)):
        ref = K.polyint(K.PGAUSS, np.array([0.7]), xp, yp, 3, xc, yc, 4000)
        assert abs(K.polyint_gauss_sgl(0.7, xp, yp, 3, xc, yc, gx, gw) - ref) < 1e-7, (xc, yc)


def small():
    d = readata(ROOT / "data/jma/jap.in")
    i0 = int(np.searchsorted(d["zz"], d["tstart"])) - 100
    sl = slice(i0, i0 + 200)
    return d, sl, Catalog(*(d[k][sl].copy() for k in ("xx", "yy", "zz", "zmg"))), Region(
        d["tx"], d["ty"], d["tstart"], d["zz"][sl][-1])


def test_gradient_and_threads():
    d, sl, cat, reg = small()
    sd = StochasticDeclustering(cat, reg, ETASParams.from_array(d["b"]), Settings(threads=2))
    sd.settings.use_threads()
    sd.compute_bandwidths()
    args = (cat.x, cat.y, sd.background_prob, sd.bandwidth, reg.polygon_x, reg.polygon_y, reg.npoly,
            reg.t_end, reg.t_start)
    u1, u2 = np.empty(len(cat)), np.empty(len(cat))
    gb = K.gauss_nodes(64)
    x1 = K.bkgdcalc_sgl(*args, u1, *gb)
    x2 = K.bkgdcalc_sgl_par(*args, u2, *gb)
    assert x1 == x2 and np.array_equal(u1, u2)

    b = np.sqrt(np.array(d["b"], dtype=float))
    g1, g2 = np.empty(8), np.empty(8)
    nodes = K.gauss_nodes(48)
    tail = (sd.ind, u1, x1, cat.x, cat.y, cat.t, cat.m, reg.polygon_x, reg.polygon_y, reg.npoly,
            reg.t_end, reg.t_start)
    f1 = K.func15_sgl(b, *tail, g1, *nodes)
    f2 = K.func15_sgl_par(b, *tail, g2, *nodes)
    assert f1 == f2 and np.array_equal(g1, g2)
    for k in range(8):
        e = 1e-6 * b[k]
        bp, bm = b.copy(), b.copy()
        bp[k] += e
        bm[k] -= e
        fd = (K.func15_sgl(bp, *tail, np.empty(8), *nodes)
              - K.func15_sgl(bm, *tail, np.empty(8), *nodes)) / (2 * e)
        assert abs(g1[k] - fd) < 1e-5 * max(1.0, abs(fd)), (k, g1[k], fd)


def test_cheaper_lambda_sums():
    """cache and value-only: same bits as xlamb. logs: rounding only, and its gradient is
    the gradient of its own value. Serial == parallel for both."""
    d, _, cat, reg = small()
    sd = StochasticDeclustering(cat, reg, ETASParams.from_array(d["b"]), Settings(threads=2))
    sd.settings.use_threads()
    sd.compute_bandwidths()
    args = (cat.x, cat.y, sd.background_prob, sd.bandwidth, reg.polygon_x, reg.polygon_y, reg.npoly,
            reg.t_end, reg.t_start)
    u = np.empty(len(cat))
    xint0 = K.bkgdcalc_sgl(*args, u, *K.gauss_nodes(64))
    b = np.sqrt(np.array(d["b"], dtype=float))
    a = (cat.x, cat.y, cat.t, cat.m)
    terms = K._parent_terms(b, cat.m)
    for i in range(1, len(cat), 17):
        g0, g1, g2 = np.empty(8), np.empty(8), np.empty(8)
        v0 = K.xlamb(i, b, u, *a, g0)
        assert K.xlamb_cache(i, b, u, *a, g1, *terms) == v0 and np.array_equal(g0, g1)
        assert K._xlamb_value(i, b, u, cat.x, cat.y, cat.t, *terms) == v0
        assert abs(K.xlamb_logs(i, b, u, *a, g2, *terms) / v0 - 1) < 1e-14
        assert np.max(np.abs(g2 - g0) / (np.abs(g0) + 1e-300)) < 1e-10
    p0, p1, p2 = np.empty(len(cat)), np.empty(len(cat)), np.empty(len(cat))
    K.probcalc(b, u, *a, p0)
    K.probcalc_value(b, u, *a, p1)
    K.probcalc_value_par(b, u, *a, p2)
    assert np.array_equal(p0, p1) and np.array_equal(p0, p2)

    nodes = K.gauss_nodes(48)
    tail = (sd.ind, u, xint0, *a, reg.polygon_x, reg.polygon_y, reg.npoly, reg.t_end, reg.t_start)
    for logs in (True, False):
        g1, g2 = np.empty(8), np.empty(8)
        f1 = K.func15_sgl(b, *tail, g1, *nodes, 1e-25, 100.0, logs)
        f2 = K.func15_sgl_par(b, *tail, g2, *nodes, 1e-25, 100.0, logs)
        assert f1 == f2 and np.array_equal(g1, g2)
        for k in range(8):
            e = 1e-6 * b[k]
            bp, bm = b.copy(), b.copy()
            bp[k] += e
            bm[k] -= e
            fd = (K.func15_sgl(bp, *tail, np.empty(8), *nodes, 1e-25, 100.0, logs)
                  - K.func15_sgl(bm, *tail, np.empty(8), *nodes, 1e-25, 100.0, logs)) / (2 * e)
            assert abs(g1[k] - fd) < 1e-5 * max(1.0, abs(fd)), (logs, k, g1[k], fd)


def test_settings_and_agreement_with_original_quadrature():
    assert Settings().quadrature == "sgl" and Settings().xlamb == "logs"
    assert config.PRESETS["original"]["quadrature"] == "ndiv"
    for bad in (dict(quadrature="x"), dict(sgl_nodes=0), dict(sgl_nodes_background=0), dict(xlamb="x")):
        try:
            Settings(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    d, _, cat, reg = small()
    fits = {}
    for q in ("sgl", "ndiv"):
        fits[q] = StochasticDeclustering(
            cat, reg, ETASParams.from_array(d["b"]),
            Settings(schedule=(1,), converge=False, ndiv_check=False, quadrature=q)).fit()
    assert np.max(np.abs(fits["sgl"].params.to_array() / fits["ndiv"].params.to_array() - 1)) < 1e-3
    assert np.max(np.abs(fits["sgl"].background_prob - fits["ndiv"].background_prob)) < 1e-4


if __name__ == "__main__":
    test_etas_kernel_and_integrands_against_closed_form()
    test_q_integrand_against_finite_difference()
    test_gaussian()
    test_gradient_and_threads()
    test_cheaper_lambda_sums()
    test_settings_and_agreement_with_original_quadrature()
    print("ok")
