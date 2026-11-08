"""geo: units, polygon file/checks, projections.
Run: python -I tests/test_geo.py"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy import geo  # noqa: E402
from steqpy.run import readata  # noqa: E402

# JMA polygon from jap.in (lon lat, counter-clockwise)
JMA = np.array([(134.0, 31.9), (137.9, 33.0), (143.1, 33.2), (144.9, 35.2), (147.8, 41.3),
                (137.8, 44.2), (137.4, 40.2), (135.1, 38.0), (130.6, 35.4)])


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def test_units():
    assert geo.length_deg("0.05 deg") == 0.05
    assert abs(geo.length_deg(f"{geo.KM_PER_DEG} km") - 1.0) < 1e-15
    assert abs(geo.area_deg2(f"{geo.KM_PER_DEG ** 2} km2") - 1.0) < 1e-15
    assert raises(geo.length_deg, 0.05)            # bare number: units are mandatory
    assert raises(geo.length_deg, "0.05 m")
    assert raises(geo.area_deg2, "0.05 deg")


def test_polygon_file_and_checks():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "poly.txt"
        p.write_bytes(("# JMA\n" + "\n".join(f"{a}, {b}" for a, b in JMA)
                       + f"\n{JMA[0, 0]} {JMA[0, 1]}\n\n").encode("utf-8"))
        lon, lat = geo.check_lonlat_polygon(*geo.read_polygon(p))
    assert np.array_equal(lon, JMA[:, 0]) and np.array_equal(lat, JMA[:, 1])

    x, y = geo.check_planar_polygon(lon, lat)      # CCW: unchanged, closed
    assert np.array_equal(x[:-1], lon) and x[-1] == x[0] and y[-1] == y[0]
    msgs = []
    xr, yr = geo.check_planar_polygon(lon[::-1], lat[::-1], msgs.append)
    assert msgs and np.array_equal(xr[:-1], lon[::-1][::-1])   # reversed back to CCW

    sq = np.array([0, 1, 1, 0.0]), np.array([0, 0, 1, 1.0])
    assert not raises(geo.check_planar_polygon, *sq)
    assert raises(geo.check_planar_polygon, np.array([0, 1, 0, 1.0]), sq[1])   # bow-tie
    assert raises(geo.check_planar_polygon, sq[0][:2], sq[1][:2])             # < 3
    assert raises(geo.check_planar_polygon, np.array([0, 1, 2.0]), np.zeros(3))  # no area
    assert raises(geo.check_planar_polygon, np.array([0, 1, 1, 1, 0.0]),
                  np.array([0, 0, 1, 1, 1.0]))                               # repeated
    assert raises(geo.check_lonlat_polygon, np.array([-179, 179, 179.0]),
                  np.array([0, 0, 1.0]))                                     # antimeridian
    assert raises(geo.check_lonlat_polygon, np.array([0, 1, 1.0]), np.array([0, 0, 91.0]))


def test_flat_matches_original():
    d = readata(ROOT / "data/jma/jap.in")
    pr = geo.Projection("flat", JMA[:, 0], JMA[:, 1])
    x, y = pr(JMA[:, 0], JMA[:, 1])
    assert np.array_equal(x, d["tx"][:-1]) and np.array_equal(y, d["ty"][:-1])


def test_laea():
    pr = geo.Projection("laea", JMA[:, 0], JMA[:, 1])
    x0, y0 = pr(pr.lon0, pr.lat0)
    assert abs(x0) < 1e-12 and abs(y0) < 1e-12
    # distances: radial scale cos(rho/2), tangential 1/cos(rho/2) -> small within ~10 deg
    rng = np.random.default_rng(0)
    lon = pr.lon0 + rng.uniform(-8, 8, 400)
    lat = pr.lat0 + rng.uniform(-6, 6, 400)
    x, y = pr(lon, lat)
    i, j = rng.integers(0, 400, (2, 2000))
    gc = geo.great_circle_deg(lon[i], lat[i], lon[j], lat[j])
    ok = (gc > 0.01) & (gc < 1.0)
    ratio = np.hypot(x[i] - x[j], y[i] - y[j])[ok] / gc[ok]
    assert np.all(np.abs(ratio - 1) < 0.01), (ratio.min(), ratio.max())
    # equal area: a 0.1 x 0.1 deg cell at lat 40 (spherical area in deg^2 of arc)
    lo, la = np.array([140, 140.1, 140.1, 140]), np.array([40, 40, 40.1, 40.1])
    xs, ys = pr(lo, la)
    a_proj = 0.5 * abs(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))
    r = 180 / math.pi
    a_true = r * r * math.radians(0.1) * (math.sin(math.radians(40.1))
                                          - math.sin(math.radians(40)))
    assert abs(a_proj / a_true - 1) < 1e-5, a_proj / a_true


def test_inverse():
    """inverse undoes the projection: both kinds, JMA and a southern-hemisphere polygon
    (Indonesia), points up to ~20 deg from the centre."""
    rng = np.random.default_rng(1)
    indo = np.array([[95.78, -15.40], [135.77, -15.40], [135.77, 1.91], [95.78, 1.91]])
    for poly in (JMA, indo):
        for kind in ("flat", "laea"):
            pr = geo.Projection(kind, poly[:, 0], poly[:, 1])
            lon = pr.lon0 + rng.uniform(-20, 20, 500)
            lat = np.clip(pr.lat0 + rng.uniform(-20, 20, 500), -80, 80)
            lo, la = pr.inverse(*pr(lon, lat))
            assert np.max(np.abs(lo - lon)) < 1e-9 and np.max(np.abs(la - lat)) < 1e-9, kind
            lo, la = pr.inverse(*pr(pr.lon0, pr.lat0))      # centre: rho = 0
            assert abs(lo - pr.lon0) < 1e-12 and abs(la - pr.lat0) < 1e-12


if __name__ == "__main__":
    test_units()
    test_polygon_file_and_checks()
    test_flat_matches_original()
    test_laea()
    test_inverse()
    print("ok")
