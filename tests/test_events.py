"""EventCatalog (ingest), completeness methods, StochasticDeclustering.from_catalog.
- CSV: header aliases, any column order, missing depth, UTC times, sorted, bad rows named;
- from_catalog builds exactly the arrays from_config builds from the same data and settings;
- write_outputs saves an in-memory catalogue so the written config.toml reproduces the inputs;
- Mc methods find a known roll-off on a synthetic catalogue; b near the true value.
Run: python -I tests/test_events.py"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy import EventCatalog, StochasticDeclustering, completeness  # noqa: E402
from steqpy.declustering import DEFAULT_START  # noqa: E402

TOML = ROOT / "configs/jma.toml"
JMA = ROOT / "data/jma/alljpM45.etas"


def test_csv():
    text = ("Mag,Origin Time (UTC),Lat,Lon,Depth (km),agency\n"
            "5.2,2016-09-12 11:32:54,35.76,129.19,15.0,KMA\n"
            "5.1,2016-09-12T10:44:32Z,35.77,129.18,,KMA\n"
            "\n"
            "3.0,2016-09-13,35.70,129.20,12.5,KMA\n")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.csv"
        p.write_text(text, encoding="utf-8")
        e = EventCatalog.from_csv(p)
        assert len(e) == 3 and str(e.t0) == "2016-09-12T10:44:32.000"
        assert np.allclose(e.t, [0, 2902 / 86400, 47728 / 86400], rtol=0, atol=1e-12)
        assert e.mag.tolist() == [5.1, 5.2, 3.0] and e.lon.tolist() == [129.18, 129.19, 129.20]
        assert np.isnan(e.depth[0]) and e.depth[1] == 15.0
        assert e.days("2016-09-12 10:44:32") == 0.0 and e.days(2.5) == 2.5
        assert len(e.between("2016-09-12 11:00", None)) == 2
        for bad in ("Mag,Lat,Lon\n5,1,2\n",                                   # no time column
                    "time,lon,lat,mag\n2016-01-01,200,1,5\n",                 # lon out of range
                    "time,lon,lat,mag\nyesterday,1,1,5\n",                    # unreadable time
                    "time,lon,lat,mag\n2016-01-01T00:00+09:00,1,1,5\n"):      # offsets refused
            p.write_text(bad, encoding="utf-8")
            try:
                EventCatalog.from_csv(p)
                raise AssertionError(f"accepted: {bad!r}")
            except ValueError:
                pass


def test_from_catalog_matches_config():
    """Same data, polygon, window, cutoff, start values and jitter: same arrays, both paths."""
    poly = np.loadtxt(ROOT / "configs/jma_polygon.txt").tolist()
    for jit in (None, {"lon": 0.01, "mag": "precision"}):
        raw = {"input": {"catalog": str(JMA), "polygon": poly, "t_min": 0.0, "t_start": 10000.0,
                         "t_end": 23376.0, "mag_min": 4.5},
               "params": DEFAULT_START, "run": {"bootstrap_seed": 4}}       # from_catalog: seed=4 for all
        if jit:
            raw["input"].update({f"jitter_{k}": v for k, v in jit.items()}, jitter_seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = StochasticDeclustering.from_config(raw)
            b = StochasticDeclustering.from_catalog(EventCatalog.from_etas(JMA), poly, 10000.0, 23376.0,
                                                    4.5, t_min=0.0, jitter=jit, seed=4)
        for f in ("x", "y", "t", "m"):
            assert np.array_equal(getattr(a.catalog, f), getattr(b.catalog, f)), (jit, f)
        assert np.array_equal(a.ind, b.ind) and a.settings == b.settings


def test_outputs_reproduce_inputs():
    """write_outputs saves the in-memory catalogue; its config.toml rebuilds the same arrays."""
    rng = np.random.default_rng(0)
    n = 400
    e = EventCatalog.from_arrays(rng.uniform(140, 142, n), rng.uniform(36, 38, n),
                                 np.datetime64("2000-01-01") + rng.integers(0, 3650 * 86400, n).astype("timedelta64[s]"),
                                 np.round(4 + rng.exponential(0.43, n), 1))
    poly = [[140.3, 36.3], [141.7, 36.3], [141.7, 37.7], [140.3, 37.7]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = StochasticDeclustering.from_catalog(e, poly, "2003-01-01", "2009-12-31", 4.0,
                                                 jitter={"lon": "precision", "lat": "precision"}, seed=2,
                                                 max_outer=2)
        with tempfile.TemporaryDirectory() as tmp:
            sd.fit().write_outputs(tmp)
            assert (Path(tmp) / "catalogue.etas").exists()
            sd2 = StochasticDeclustering.from_config(Path(tmp) / "config.toml")
    for f in ("x", "y", "t", "m"):
        assert np.array_equal(getattr(sd.catalog, f), getattr(sd2.catalog, f)), f
    assert sd.settings.bootstrap_seed == 2 and sd.config["input"]["jitter_seed"] == 2   # one seed for all
    r = sd.result()
    assert np.array_equal(r.sample_background(), r.sample_background())             # seeded by default
    assert r.to_dict()["seeds"] == {"jitter": 2, "bootstrap": 2}
    # Mc over time: a catalogue complete above 4.0 for 10 years, then above 3.0
    rng = np.random.default_rng(3)
    m = np.round(2.5 + rng.exponential(np.log10(np.e), 60_000), 1)
    t = rng.uniform(0, 7305, m.size)
    keep = m >= np.where(t < 3652, 4.0, 3.0)
    e = EventCatalog.from_arrays(np.zeros(keep.sum()), np.zeros(keep.sum()), t[keep], m[keep])
    s = e.mc_over_time("maxc")
    early, late = s["mc"][s["t"] < 3500], s["mc"][s["t"] > 3900]
    early, late = early[np.isfinite(early)], late[np.isfinite(late)]          # NaN: < 50 events
    assert early.size > 5 and np.all(early >= 4.0) and np.all(late <= 3.4), (early, late)
    assert s["counts"].sum() == len(e) and len(s["t"]) == len(s["counts"]) - 1


def test_completeness():
    """GR with b = 1 above 3.0, detection falling off below it: methods land near 3.0."""
    rng = np.random.default_rng(1)
    m = 2.0 + rng.exponential(np.log10(np.e), 200_000)                 # b = 1 from 2.0
    seen = rng.random(m.size) < np.clip(1 - (3.0 - m) / 1.0, 0, 1) ** 3  # roll-off below 3.0
    m = np.round(m[seen], 1)
    e = EventCatalog(np.zeros(m.size), np.zeros(m.size), np.arange(m.size, dtype=float), m)
    got = {k: e.completeness(k) for k in completeness.METHODS}
    print("Mc:", got)
    for k, v in got.items():
        assert 2.8 <= v <= 3.4, (k, v)
    b, se, nn = e.b_value(3.2)
    assert abs(b - 1) < 3 * se + 0.02, (b, se)
    assert np.isnan(completeness.mc(m[:10], "maxc"))                   # too few to judge


if __name__ == "__main__":
    test_csv()
    test_completeness()
    test_from_catalog_matches_config()
    test_outputs_reproduce_inputs()
    print("ok")
