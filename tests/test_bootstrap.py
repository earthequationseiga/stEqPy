"""Parametric bootstrap: the simulator's distributions, and a small end-to-end run.
Run: python tests/test_bootstrap.py"""
import dataclasses
import math
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from steqpy import bootstrap  # noqa: E402
from test_outputs import small_fit  # noqa: E402


def test_offspring_distributions():
    """Medians of the time and squared-distance offsets against the closed forms
    c (2^(1/(p-1)) - 1) and s (2^(1/(q-1)) - 1); mean number of children A e^(alpha m)."""
    th = np.array([0.5, 0.3, 0.03, 1.2, 1.15, 0.002, 1.9, 1.0])
    rng = np.random.default_rng(0)
    n = 200000
    m = np.full(n, 0.5)
    t, x, y, mm = bootstrap.offspring(np.zeros(n), np.zeros(n), np.zeros(n), m, th, rng, np.array([0.0]))
    kids = th[1] * math.exp(th[3] * 0.5)
    assert abs(t.size / n / kids - 1) < 0.01, t.size / n
    # Half the draws fall below each closed-form median, within 5 sigma of a binomial fraction.
    tol = 5 * 0.5 / math.sqrt(t.size)
    assert abs(np.mean(t < th[2] * (2 ** (1 / (th[4] - 1)) - 1)) - 0.5) < tol
    s = th[5] * math.exp(th[7] * 0.5)
    assert abs(np.mean(x ** 2 + y ** 2 < s * (2 ** (1 / (th[6] - 1)) - 1)) - 0.5) < tol
    assert np.all(mm == 0.0)


def test_simulate_and_run():
    """Background-only simulation (A tiny, so no offspring): one declustering draw, so the count
    is binomial with mean sum(phi), each event keeps its own time and magnitude. With the fitted
    A, offspring stay inside the catalogue's bounding box. Then a 3-replicate bootstrap on a
    small fit: shapes, output files."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = small_fit()
    th = res.params.to_array()
    th[1] = 1e-12
    quiet = dataclasses.replace(res, params=type(res.params).from_array(th))
    phi = quiet.final_background_prob
    rng = np.random.default_rng(1)
    sims = [bootstrap.simulate(quiet, rng) for _ in range(200)]
    counts = np.array([len(c) for c in sims])
    sd = math.sqrt(np.sum(phi * (1 - phi)) / 200)
    assert abs(counts.mean() - phi.sum()) < 5 * max(sd, 1e-9), (counts.mean(), phi.sum())
    cat = res.catalog
    for c in sims:                  # times are unique in the catalogue, so t identifies the event
        i = np.searchsorted(cat.t, c.t)
        assert np.array_equal(cat.t[i], c.t) and np.array_equal(cat.m[i], c.m)
    full_sims = [bootstrap.simulate(res, rng) for _ in range(20)]
    assert sum(len(c) for c in full_sims) > counts[:20].sum()          # offspring did occur
    off = np.concatenate([c.x[~np.isin(c.t, cat.t)] for c in full_sims])
    offy = np.concatenate([c.y[~np.isin(c.t, cat.t)] for c in full_sims])
    assert off.size and off.min() >= cat.x.min() and off.max() <= cat.x.max()
    assert offy.min() >= cat.y.min() and offy.max() <= cat.y.max()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = bootstrap.run(res, 3, seed=2)
    assert b["params"].shape == (3, 8) and b["u"].shape == (len(res.catalog),)
    print("bootstrap good replicates:", b["n_good"], "se:", b["se"])
    full = dataclasses.replace(res, bootstrap=b)
    with tempfile.TemporaryDirectory() as tmp:
        full.write_outputs(tmp)
        assert np.loadtxt(Path(tmp) / "bootstrap_params.dat").shape == (3, 8)
        if b["n_good"] > 1:
            txt = (Path(tmp) / "params.txt").read_text()
            assert "parametric bootstrap" in txt and "95% (" in txt


if __name__ == "__main__":
    test_offspring_distributions()
    test_simulate_and_run()
    print("ok")
