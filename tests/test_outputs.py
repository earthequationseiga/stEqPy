"""Declustering outputs: triggering probabilities and the two samplers.
Run: python tests/test_outputs.py"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steqpy import Catalog, ETASParams, Region, Settings, StochasticDeclustering  # noqa: E402
from steqpy.run import readata  # noqa: E402

N_DRAWS = 2000
TOL = 5 * 0.5 / np.sqrt(N_DRAWS)    # 5 sigma of a mean of N_DRAWS Bernoulli draws, worst case


def small_fit(**kw):
    # t_end must match the slice: over the full window, 200 events would drive mu to ~0
    # and every phi to 0 or 1, and the sampler tests would have nothing to check
    d = readata(ROOT / "data/jma/jap.in")
    i0 = int(np.searchsorted(d["zz"], d["tstart"])) - 100
    sl = slice(i0, i0 + 200)
    return StochasticDeclustering(
        Catalog(*(d[k][sl].copy() for k in ("xx", "yy", "zz", "zmg"))),
        Region(d["tx"], d["ty"], d["tstart"], d["zz"][sl][-1]),
        ETASParams.from_array(d["b"]),
        Settings(schedule=(1,), converge=False, ndiv_check=False, ndiv=50, **kw),
    ).fit()


def test_probabilities_sum_to_one(res):
    phi = res.final_background_prob
    for i in range(len(phi)):
        rho = res.triggering_probs(i)
        assert len(rho) == i and np.all(rho >= 0)
        assert abs(phi[i] + rho.sum() - 1) < 1e-12, i
    for bad in (-1, len(phi)):
        try:
            res.triggering_probs(bad)
        except IndexError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_final_probs_use_final_params(res):
    # the MLE moved the parameters, so these differ from the pre-MLE probs.dat values
    assert np.max(np.abs(res.final_background_prob - res.background_prob)) > 1e-6
    # Same kernel (A form): same bits. K form fit, A form outputs: same model, rounding only.
    frozen = small_fit(optimise=False, form="A", optimiser="davidn")
    assert np.array_equal(frozen.final_background_prob, frozen.background_prob)
    frozen = small_fit(optimise=False)
    assert np.max(np.abs(frozen.final_background_prob - frozen.background_prob)) < 1e-14


def test_seeds_and_threads(res):
    a, b = res.sample_background(seed=7), res.sample_background(seed=7)
    assert a.dtype == bool and np.array_equal(a, b)
    assert not np.array_equal(a, res.sample_background(seed=8))
    p = res.sample_parents(seed=7)
    assert np.array_equal(p, res.sample_parents(seed=7))
    assert not np.array_equal(p, res.sample_parents(seed=8))
    threaded = dataclasses.replace(res, settings=dataclasses.replace(res.settings, threads=2))
    assert np.array_equal(p, threaded.sample_parents(seed=7))
    assert np.all((p >= -1) & (p < np.arange(len(p))))     # parents are earlier events
    assert np.array_equal(a, p == -1)                      # same seed, same background events


def test_draws_converge_to_probabilities(res):
    phi = res.final_background_prob
    kept = np.mean([res.sample_background(seed=s) for s in range(N_DRAWS)], axis=0)
    assert np.max(np.abs(kept - phi)) < TOL
    parents = np.array([res.sample_parents(seed=s) for s in range(N_DRAWS)])
    assert np.max(np.abs(np.mean(parents == -1, axis=0) - phi)) < TOL
    i = int(np.argmin(phi))                                 # the most clearly triggered event
    freq = np.bincount(parents[:, i][parents[:, i] >= 0], minlength=i) / N_DRAWS
    assert np.max(np.abs(freq - res.triggering_probs(i))) < TOL
    print(f"max |mean draw - phi| = {np.max(np.abs(kept - phi)):.4f}, tolerance {TOL:.4f}")


def test_k_form_outputs(res):
    """K-form fit: outputs equal the A-form ones at p, q > 1 (rounding only), and exist for
    p, q <= 1, where A is undefined."""
    assert res.K is not None
    as_a = dataclasses.replace(res, K=None)         # same params through the A-form kernels
    gx, gy = np.linspace(res.catalog.x.min(), res.catalog.x.max(), 4), np.linspace(
        res.catalog.y.min(), res.catalog.y.max(), 3)
    assert np.max(np.abs(res.final_background_prob - as_a.final_background_prob)) < 1e-13
    ra, rk = as_a.rates(gx, gy), res.rates(gx, gy)
    assert np.max(np.abs(rk / ra - 1)) < 1e-12
    i = len(res.catalog) - 1
    assert np.max(np.abs(res.triggering_probs(i) - as_a.triggering_probs(i))) < 1e-13

    th = res.params.to_array()
    th[1], th[4], th[6] = np.nan, 0.9, 0.95
    low = dataclasses.replace(res, params=ETASParams.from_array(th))
    phi = low.final_background_prob
    assert np.all(np.isfinite(phi)) and np.all((phi > 0) & (phi <= 1))
    for i in range(1, len(phi), 23):
        rho = low.triggering_probs(i)
        assert np.all(rho >= 0) and abs(phi[i] + rho.sum() - 1) < 1e-12
    assert np.all(np.isfinite(low.rates(gx, gy)))
    pa = low.sample_parents(seed=3)
    assert np.all((pa >= -1) & (pa < np.arange(len(pa))))


def test_result_json(res):
    """result.json: strict JSON (no NaN), summary bands and histogram add up over the targets,
    per-event values match the result, background map = mu * u on the grid, events and
    polygon mapped back to lon/lat (flat projection of the JMA polygon, as readata uses)."""
    import json
    import tempfile
    from steqpy import geo
    plon, plat = geo.read_polygon(str(ROOT / "configs/jma_polygon.txt"))
    proj = geo.Projection("flat", plon, plat)
    full = dataclasses.replace(res, projection=proj,
                               grid={"lon": [138.0, 142.0], "lat": [34.0, 38.0], "n": [5, 4]})
    with tempfile.TemporaryDirectory() as tmp:
        full.write_outputs(tmp)
        d = json.loads(Path(tmp, "result.json").read_text(encoding="utf-8"),
                       parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    s, e, phi = d["summary"], d["events"], res.final_background_prob
    tg = res.ind == 1
    assert s["n_targets"] == tg.sum() == d["fit"]["n_targets"]
    assert s["certain_background"] + s["uncertain"] + s["certain_aftershock"] == s["n_targets"]
    assert sum(s["histogram_p_aftershock"]["counts"]) == s["n_targets"]
    assert s["certain_aftershock"] == np.sum(1 - phi[tg] > 0.9)
    assert abs(s["expected_background"] - phi[tg].sum()) < 1e-9
    assert np.allclose(e["p_background"], phi, rtol=1e-5, atol=1e-12) and e["target"] == list(tg.astype(int))
    lon, lat = np.array(e["lon"]), np.array(e["lat"])
    x, y = proj(lon, lat)
    assert np.max(np.abs(x - res.catalog.x)) < 1e-5 and np.max(np.abs(y - res.catalog.y)) < 1e-5
    assert len(d["region"]["polygon"]) == len(res.region.polygon_x)
    # The background map as a viewer draws it, from the JSON alone, against steqpy's kernel
    # (given the same, final phi).
    assert d["projection"]["kind"] == "flat" and "background_map" not in d
    b = d["background"]
    glon, glat = np.meshgrid(np.linspace(138, 142, 7), np.linspace(34, 38, 6), indexing="ij")
    gx, gy = proj(glon.ravel(), glat.ravel())
    ex, ey = proj(lon, lat)
    h, ph = np.array(e["bandwidth"]), np.array(e["p_background"])
    r2 = (gx[:, None] - ex) ** 2 + (gy[:, None] - ey) ** 2
    viewer = b["mu"] / b["T_days"] * np.sum(ph * np.exp(-r2 / (2 * h * h)) / (2 * np.pi * h * h), axis=1)
    same_phi = dataclasses.replace(full, background_prob=phi)
    steqpy_pts = res.params.mu * same_phi.rates_lonlat(glon.ravel(), glat.ravel())[:, 0]
    assert np.allclose(viewer, steqpy_pts, rtol=1e-5), np.max(np.abs(viewer / steqpy_pts - 1))
    assert d["fit"]["valid"] in (True, False) and set(d["params"]) >= {"mu", "A", "p", "q"}
    print("result.json:", {k: s[k] for k in ("certain_background", "uncertain", "certain_aftershock")})


if __name__ == "__main__":
    fit = small_fit()
    undecided = int(np.sum((fit.final_background_prob > 0.05) & (fit.final_background_prob < 0.95)))
    print("events with 0.05 < phi < 0.95:", undecided)
    assert undecided >= 20
    test_probabilities_sum_to_one(fit)
    test_final_probs_use_final_params(fit)
    test_seeds_and_threads(fit)
    test_draws_converge_to_probabilities(fit)
    test_k_form_outputs(fit)
    test_result_json(fit)
    print("ok")
