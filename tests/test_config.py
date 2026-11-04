"""Config loading, presets, validation, convergence loop.
- control file and TOML (polygon text file, explicit inputs) build exactly the arrays
  readata builds (so the "original" preset stays bit-exact);
- bad configs fail loudly; resolved config round-trips;
- converge=True stops on its tolerances; ndiv check runs.
Run: python -I tests/test_config.py"""
import dataclasses
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from etas8py import config  # noqa: E402
from etas8py.declustering import (Catalog, ETASParams, Region, Settings,  # noqa: E402
                                  StochasticDeclustering)
from etas8py.run import readata  # noqa: E402

JAP = ROOT / "data/jma/jap.in"
TOML = ROOT / "configs/jma.toml"


def same_as_readata(sd, d):
    c, r = sd.catalog, sd.region
    for a, b in ((c.x, d["xx"]), (c.y, d["yy"]), (c.t, d["zz"]), (c.m, d["zmg"]),
                 (sd.ind, d["ind"]), (r.polygon_x, d["tx"]), (r.polygon_y, d["ty"])):
        assert np.array_equal(a, b)
    assert r.t_start == d["tstart"] and r.t_end == d["tz"]
    assert np.array_equal(sd.params.to_array(), d["b"])
    assert sd.settings.npp == d["npp"] and sd.settings.bwm_deg == d["bwm"]


def raises(fn, **kw):
    try:
        fn(**kw)
    except ValueError:
        return True
    return False


def test_inputs_match_readata():
    d = readata(JAP)
    sd = StochasticDeclustering.from_input_file(JAP, preset="original")
    same_as_readata(sd, d)
    s = sd.settings
    assert not s.converge and not s.fast_xint and not s.ndiv_check
    sd2 = StochasticDeclustering.from_config(TOML, preset="original")   # explicit inputs
    same_as_readata(sd2, d)
    lon, lat, r = sd2.grid["lon"], sd2.grid["lat"], d["grid"]
    assert sd2.grid["n"] == [r[0], r[1]] and lat == [r[4], r[5]]
    assert np.allclose(np.array(lon) * sd2.projection.cosm, r[2:4], rtol=0, atol=1e-13)
    assert StochasticDeclustering.from_config(TOML).settings.converge   # TOML: default


def test_reversed_polygon_and_overrides():
    poly = np.loadtxt(ROOT / "configs/jma_polygon.txt")
    base = {"input": {"control": str(JAP), "polygon": poly[::-1].tolist()}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sd = StochasticDeclustering.from_config(base, preset="original")
    assert any("clockwise" in str(x.message) for x in w)
    same_as_readata(sd, readata(JAP))
    sd = StochasticDeclustering.from_config(base, npp=7, bwm="11.1195 km", threads=2)
    assert sd.settings.npp == 7 and abs(sd.settings.bwm_deg - 0.1) < 1e-6


def test_bad_configs():
    ctl = {"control": str(JAP)}
    f = StochasticDeclustering.from_config
    assert raises(f, source={"input": ctl, "model": {"npps": 5}})          # typo
    assert raises(f, source={"input": ctl, "model": {"bwm": 0.05}})        # no units
    assert raises(f, source={"input": ctl, "params": {"D": 0.0012}})       # no units
    assert raises(f, source={"input": ctl}, threads=100000)
    assert raises(f, source={"input": ctl}, projection="mercator")
    assert raises(f, source={"input": ctl}, preset="fast")
    assert raises(f, source={"input": {"catalog": "x"}})                    # incomplete
    assert raises(f, source={"input": ctl, "polygon": [[0, 0], [1, 1]]})    # bad table


def test_sections_cover_settings_and_roundtrip():
    names = [f.name for f in dataclasses.fields(Settings)]
    flat = [k for ks in config.SECTIONS.values() for k in ks]
    assert sorted(names) == sorted(flat) and len(flat) == len(set(flat))
    sd = StochasticDeclustering.from_config(TOML, projection="laea", inner_tol=3e-5)
    with tempfile.TemporaryDirectory() as tmp:
        c = sd.config
        p = Path(tmp) / "c.toml"
        p.write_bytes(config.dump(c["settings"], c["input"], c["params"], c["grid"],
                                  c["preset"]).encode("utf-8"))
        sd2 = StochasticDeclustering.from_config(p)
    assert sd2.settings == sd.settings
    assert np.array_equal(sd2.catalog.x, sd.catalog.x)
    assert np.array_equal(sd2.params.to_array(), sd.params.to_array())


def test_convergence_small():
    d = readata(JAP)
    n = 600
    i0 = int(np.searchsorted(d["zz"], d["tstart"])) - n // 2
    sl = slice(i0, i0 + n)
    cat = Catalog(d["xx"][sl].copy(), d["yy"][sl].copy(), d["zz"][sl].copy(),
                  d["zmg"][sl].copy())
    reg = Region(d["tx"], d["ty"], d["tstart"], d["tz"])
    init = ETASParams.from_array(d["b"])
    for inner, outer in (("dphi", "loglik"), ("loglik", "dphi")):
        s = Settings(npp=d["npp"], max_outer=6, inner_metric=inner, outer_metric=outer,
                     inner_tol=1e-3 if inner == "dphi" else 1e-6,
                     outer_tol=1e-6 if outer == "loglik" else 1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd = StochasticDeclustering(cat, reg, init, s).fit()
        outers = [h for h in sd.history if h[0] == "outer"]
        inners = [h for h in sd.history if h[0] == "inner"]
        assert sd.ndiv_error is not None and sd.ndiv_error < 1e-6, sd.ndiv_error
        assert outers[0][3] == np.inf and len(outers) <= 6
        assert all(h[2] <= s.max_inner for h in inners)
        if sd.converged:
            assert outers[-1][3] < s.outer_tol
        print(inner, outer, "outer changes", [f"{h[3]:.1e}" for h in outers],
              "passes", [max(h[2] for h in inners if h[1] == k) for k, *_ in
                         [(o[1],) for o in outers]], "converged", sd.converged)


if __name__ == "__main__":
    test_inputs_match_readata()
    test_reversed_polygon_and_overrides()
    test_bad_configs()
    test_sections_cover_settings_and_roundtrip()
    test_convergence_small()
    print("ok")
