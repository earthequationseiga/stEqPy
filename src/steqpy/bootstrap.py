"""Parametric bootstrap: simulate catalogues from a fitted model, refit each, and use the spread.

Optional (Settings.bootstrap = number of replicates, 0 = off) because it costs one full fit per
replicate. Events are not independent (aftershocks depend on their parents), so resampling
events would break the clustering; simulating from the fit keeps it. Unlike the Hessian errors,
the spread includes the background's own uncertainty, because each replicate re-estimates it.

Follows Algorithm 1 of Dutilleul, Genest & Peng (2024, GJI 236, 1601-1608) as implemented in
their R package ETASbootstrap 0.2.1 (functions simulate_background_earthquakes and
simulate_aftershocks), adapted from Algorithm C of Zhuang et al. (2004). One catalogue from a
fit, over [0, t_end]:
- background: one stochastic declustering draw of the fitted catalogue (event i kept with
  probability phi_i at the final parameters); the kept events keep their times and magnitudes,
  their locations are shuffled among them (location independent of time) and each gets a
  Gaussian offset of its bandwidth d_i, so that the spatial rate matches the kernel estimate.
  (ETASbootstrap draws new background magnitudes but then returns the original column, so in
  practice they keep their own; we follow what it does.)
- offspring: each event has Poisson(A e^(alpha m)) children (the A-form kernel integrates to 1
  over infinite time and space), time offsets from the Omori density by inverse CDF,
  t = c ((1-U)^(1/(1-p)) - 1), squared distance r^2 = s ((1-U)^(1/(1-q)) - 1) with
  s = D e^(gamma m) of the parent, uniform direction; magnitudes resampled from the target
  events; generation after generation. Offspring outside the bounding box of the fitted
  catalogue (the package's longitude/latitude_boundaries default: the data's range, the edge
  zone around the study region) or after t_end are dropped and do not trigger. Events outside
  the study polygon but inside the box are kept and trigger, as auxiliary events.
  The box is in lon/lat, as in ETASbootstrap (model coordinates mapped back with
  Projection.inverse; in model coordinates when the result has no projection).
Needs p, q > 1 and a branching ratio A E[e^(alpha m)] below 1.

Results: standard deviation of the replicates and, as the paper recommends (the estimators
are skewed and biased in finite samples: on JMA and in the paper, A comes out low and c, p
high), 95% percentile intervals.
Reference: Dutilleul, P., Genest, C. & Peng, R. (2024). Bootstrapping for parameter uncertainty
in the space-time epidemic-type aftershock sequence model. GJI 236, 1601-1608. R package
ETASbootstrap 0.2.1 (Peng, Dutilleul & Genest, CRAN).
Judgement calls (differ from ETASbootstrap): replicates
that fail check_fit dropped (ETASbootstrap has no such check; the paper drops 3 outliers of
1000 from its plots, not from its intervals); refits start from the fitted parameters (as in
ETASbootstrap).
"""
import dataclasses
import math
import warnings

import numpy as np
from numba import njit, prange

from . import kernels as K
from .catalog import Catalog

MAX_FACTOR = 20         # a replicate with more than 20x the fitted catalogue's events is cut short


def offspring(t, x, y, m, th, rng, mags):
    """Children of the events (t, x, y, m): arrays (t, x, y, m), times not yet cut."""
    a, c, alpha, p, d, q, gamma = th[1], th[2], th[3], th[4], th[5], th[6], th[7]
    n = rng.poisson(a * np.exp(alpha * m))
    pt, px, py, pm = (np.repeat(v, n) for v in (t, x, y, m))
    k = pt.size
    dt = c * ((1.0 - rng.random(k)) ** (1.0 / (1.0 - p)) - 1.0)
    r = np.sqrt(d * np.exp(gamma * pm) * ((1.0 - rng.random(k)) ** (1.0 / (1.0 - q)) - 1.0))
    ang = 2.0 * math.pi * rng.random(k)
    return pt + dt, px + r * np.cos(ang), py + r * np.sin(ang), rng.choice(mags, k)


def simulate(result, rng):
    """One catalogue from the fitted model in result (a DeclusteringResult)."""
    c, reg = result.catalog, result.region
    th = result.params.to_array()
    if not (th[4] > 1 and th[6] > 1 and np.isfinite(th[1])):
        raise ValueError("bootstrap needs p > 1 and q > 1 (the A-form kernel)")
    mags = c.m[result.ind == 1]
    ratio = th[1] * np.mean(np.exp(th[3] * mags))
    if not ratio < 1:
        raise ValueError(f"branching ratio {ratio:.3g} >= 1: simulated catalogues would explode")
    h, t_end = result.bandwidth, reg.t_end
    kept = np.flatnonzero(rng.random(len(c)) < result.final_background_prob)
    src = rng.permutation(kept)                 # locations shuffled, times kept
    n0 = kept.size
    gen = (c.t[kept], c.x[src] + h[src] * rng.standard_normal(n0),
           c.y[src] + h[src] * rng.standard_normal(n0), c.m[kept])
    # lon/lat box of the fitted catalogue (model coordinates if the result has no projection)
    to_box = result.projection.inverse if result.projection is not None else (lambda x, y: (x, y))
    blon, blat = to_box(c.x, c.y)
    x0, x1, y0, y1 = blon.min(), blon.max(), blat.min(), blat.max()
    parts, total = [gen], n0
    while gen[0].size:
        gen = offspring(*gen, th, rng, mags)
        lon, lat = to_box(gen[1], gen[2])
        keep = ((gen[0] < t_end) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
        gen = tuple(v[keep] for v in gen)
        parts.append(gen)
        total += gen[0].size
        if total > MAX_FACTOR * len(c):
            warnings.warn(f"bootstrap replicate cut at {total} events")
            break
    t, x, y, m = (np.concatenate([p_[i] for p_ in parts]) for i in range(4))
    o = np.argsort(t, kind="stable")
    return Catalog(x[o], y[o], t[o], m[o])


@njit(parallel=True)
def background_at(px, py, xx, yy, phi, h, T):
    """Background rate u at points (px, py): sum_j phi_j g_j / T."""
    out = np.empty(px.shape[0])
    for k in prange(px.shape[0]):
        s = 0.0
        for j in range(xx.shape[0]):
            s += phi[j] * K.dgauss(math.sqrt((xx[j] - px[k]) ** 2 + (yy[j] - py[k]) ** 2), h[j])
        out[k] = s / T
    return out


def run(result, n, seed=0, log=lambda msg: None):
    """n replicates from result. Returns dict: params (n, 8) array (NaN rows: replicate failed
    its checks or did not converge), se {name: std over the good replicates}, ci {name: (2.5%,
    97.5%) percentiles}, n_good, and the background rate at the fitted catalogue's events:
    u (from the fit), u_se (std over good replicates)."""
    from .declustering import NAMES, StochasticDeclustering     # avoid an import cycle
    rng = np.random.default_rng(seed)
    c, reg = result.catalog, result.region
    T = reg.t_end - reg.t_start
    s = dataclasses.replace(result.settings, bootstrap=0, hessian_errors=False, ndiv_check=False)
    params = np.full((n, 8), np.nan)
    us = np.full((n, len(c)), np.nan)
    for k in range(n):
        cat = simulate(result, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sd = StochasticDeclustering(cat, reg, result.params, s)
                r = sd.fit()
            except ValueError as e:         # e.g. no target events
                log(f" bootstrap {k + 1}/{n}: {len(cat)} events, skipped ({e})")
                continue
        good = not r.checks["failed"] and r.converged is not False
        log(f" bootstrap {k + 1}/{n}: {len(cat)} events, -log L {r.neg_loglik:.4f}, "
            f"{'ok' if good else 'failed: ' + '; '.join(r.checks['failed'])}")
        if good:
            params[k] = r.params.to_array()
            us[k] = background_at(c.x, c.y, cat.x, cat.y, r.background_prob, r.bandwidth, T)
    ok = ~np.isnan(params[:, 0])
    se = (dict(zip(NAMES, (float(v) for v in np.std(params[ok], axis=0, ddof=1))))
          if ok.sum() > 1 else None)
    ci = (dict(zip(NAMES, ((float(a), float(b)) for a, b in
                           np.percentile(params[ok], [2.5, 97.5], axis=0).T)))
          if ok.sum() > 1 else None)
    u = background_at(c.x, c.y, c.x, c.y, result.background_prob, result.bandwidth, T)
    return dict(params=params, se=se, ci=ci, n_good=int(ok.sum()), u=u,
                u_se=np.std(us[ok], axis=0, ddof=1) if ok.sum() > 1 else None)
