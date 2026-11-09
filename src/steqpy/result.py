"""What a fit produces: a snapshot of the estimates, plus rate maps and output files."""
import dataclasses
import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from . import config
from . import kernels as K
from .catalog import Catalog, Region
from .params import ETASParams, Settings

# Summary bands of P(aftershock) = 1 - phi over the target events (user choice): certain
# background below 0.1, certain aftershock above 0.9, uncertain in between (both ends included).
BAND_LO, BAND_HI = 0.1, 0.9
HIST_BINS = 20
DEG_KM = 111.195                    # km per degree of great-circle arc (geo's unit)


def _clean(v):
    """JSON-ready copy: numpy types to Python, NaN and inf to None (JSON has no NaN)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else None
    return v


def _sig(a, digits=6):
    """Values rounded to `digits` significant digits (keeps result.json small)."""
    return [float(f"{x:.{digits}g}") if math.isfinite(x) else None for x in np.asarray(a, float)]


@dataclass(frozen=True, eq=False)
class DeclusteringResult:
    """Own copies of the arrays, so a later fit or step on the estimator cannot change it."""
    catalog: Catalog
    region: Region
    settings: Settings
    ind: np.ndarray                 # target flags (Region.target_flags): only 1 is a target
    params: ETASParams              # MLE, or the initial values if optimise=False
    background_prob: np.ndarray     # phi_i: P(event i is a background event)
    background_rate: np.ndarray     # u(x_i, y_i) per unit time
    bandwidth: np.ndarray           # h_i of the background kernel
    neg_loglik: float               # -log L at params
    history: list                   # (level, outer, pass, change) of the convergence tests
    converged: bool                 # None when converge=False (fixed schedule)
    n_likelihood_calls: int
    ndiv_error: float               # relative -log L change for 2*ndiv (None: not checked)
    projection: object = None       # geo.Projection, for lon/lat outputs
    grid: dict = None               # {"lon": [a, b], "lat": [a, b], "n": [.,.]}
    config: dict = None             # resolved config, when built by from_config
    checks: dict = None             # StochasticDeclustering.check_fit; "failed" lists problems
    K: float = None                 # K form: K = A (p-1)(q-1); A is NaN if p or q <= 1
    hessian_errors: dict = None     # {name: se} if Settings.hessian_errors
    bootstrap: dict = None          # bootstrap.run's output if Settings.bootstrap

    # -- declustering outputs ------------------------------------------------------------
    def _x(self):
        """sqrt(theta) for the output kernels: K in A's place for a K-form fit, so outputs
        exist for p, q <= 1 too (A is NaN there)."""
        th = self.params.to_array().copy()
        if self.K is not None:
            th[1] = self.K
        return np.sqrt(th)

    @cached_property
    def final_background_prob(self):
        """phi_i at the final parameters. background_prob (probs.dat) is from before the last
        MLE step, as in the original; the outputs below use the final parameters, as the
        original's pmatr.dat does, so that phi_i + sum_j rho_ij = 1."""
        c = self.catalog
        out = np.empty(len(c))
        self.settings.use_threads()
        if self.K is not None:
            self.settings.kernel("lambda_k")(self._x(), self.background_rate, c.x, c.y, c.t, c.m, out)
            return self.params.mu * self.background_rate / out
        self.settings.probcalc()(self._x(), self.background_rate,
                                 c.x, c.y, c.t, c.m, out)
        return out

    def triggering_probs(self, i):
        """rho_ij, j < i: the probability that earlier event j triggered event i, in an array
        of length i. Computed on demand, so all pairs need not be held in memory."""
        c = self.catalog
        if not 0 <= i < len(c):
            raise IndexError(f"event {i} out of range for {len(c)} events")
        rates = np.empty(i)
        terms = K.trigger_terms_k if self.K is not None else K.trigger_terms
        bk = terms(i, self._x(), self.background_rate, c.x, c.y, c.t, c.m, rates)
        return rates / (bk + rates.sum())

    def sample_background(self, seed=0):
        """One random declustered catalogue: a boolean mask over all events, True = background
        (Algorithm 2 of Zhuang et al. 2002). Includes non-targets: combine with
        `ind == 1` to keep only the target events. seed: fixed by default so reruns give the same
        catalogue; other seeds give other, equally valid draws."""
        phi = self.final_background_prob
        return np.random.default_rng(seed).random(len(phi)) < phi

    def sample_parents(self, seed=0):
        """One random family tree: each event's parent index, -1 = background (Algorithm 3 of
        Zhuang et al. 2004). The same seed gives the same tree at any thread count."""
        c = self.catalog
        u = np.random.default_rng(seed).random(len(c))
        self.settings.use_threads()
        return self.settings.kernel("sample_parents")(
            self._x(), self.background_rate, c.x, c.y, c.t, c.m, u, self.K is not None)

    # -- rate maps and output files ------------------------------------------------------
    def rates(self, gx, gy):
        """(len(gx)*len(gy), 3): background rate, unweighted kernel density, total
        intensity at t_end, x-major. gx, gy in model coordinates."""
        c, r = self.catalog, self.region
        self.settings.use_threads()
        return self.settings.kernel("outrates_grid")(
            self._x(), c.x, c.y, c.t, c.m, self.background_prob, self.bandwidth, r.t_end,
            r.t_start, np.asarray(gx, float), np.asarray(gy, float), self.K is not None)

    def rates_lonlat(self, lon, lat):
        """rates() at arbitrary lon/lat points (1-D arrays of equal length)."""
        c, r = self.catalog, self.region
        px, py = self.projection(lon, lat)
        self.settings.use_threads()
        return self.settings.kernel("outrates_points")(
            self._x(), c.x, c.y, c.t, c.m, self.background_prob,
            self.bandwidth, r.t_end, r.t_start, px, py, self.K is not None)

    def rate_grid(self):
        """(lon, lat, rates) on the configured lon/lat grid, x-major (lon outer)."""
        (lo1, lo2), (la1, la2), (mx, my) = self.grid["lon"], self.grid["lat"], self.grid["n"]
        lons = np.array([lo1 + (lo2 - lo1) / (mx - 1) * i for i in range(mx)])
        lats = np.array([la1 + (la2 - la1) / (my - 1) * j for j in range(my)])
        LON, LAT = (a.ravel() for a in np.meshgrid(lons, lats, indexing="ij"))
        if self.projection.kind == "flat":
            # the original's grid: spaced linearly in model x (bit-exact rates.dat)
            cm = self.projection.cosm
            tx1, tx2 = lo1 * cm, lo2 * cm
            gx = np.array([tx1 + (tx2 - tx1) / (mx - 1) * i for i in range(mx)])
            return LON, LAT, self.rates(gx, lats)
        return LON, LAT, self.rates_lonlat(LON, LAT)

    def summary(self):
        """Target events by P(aftershock) = 1 - phi (final parameters): counts in the bands
        below BAND_LO (certain background), BAND_LO..BAND_HI (uncertain) and above BAND_HI
        (certain aftershock), the expected numbers (sums of probabilities, no cutoff), and a
        histogram with HIST_BINS equal bins on [0, 1]."""
        pa = 1.0 - self.final_background_prob[self.ind == 1]
        counts, edges = np.histogram(pa, bins=HIST_BINS, range=(0.0, 1.0))
        return {"n_targets": pa.size,
                "certain_background": int(np.sum(pa < BAND_LO)),
                "uncertain": int(np.sum((pa >= BAND_LO) & (pa <= BAND_HI))),
                "certain_aftershock": int(np.sum(pa > BAND_HI)),
                "bands": [BAND_LO, BAND_HI],
                "expected_background": float(np.sum(1.0 - pa)),
                "expected_aftershocks": float(np.sum(pa)),
                "histogram_p_aftershock": {"edges": edges, "counts": counts}}

    def to_dict(self):
        """Everything a viewer (e.g. the website) needs, JSON-ready: fit status, parameters,
        errors, summary, every event, the region, and what a viewer needs to draw the
        background-rate map itself at any resolution (no grid is sent):
            mu * u(x) = mu / T * sum_j phi_j exp(-r_j^2 / (2 h_j^2)) / (2 pi h_j^2),
        r_j = distance in model coordinates (`projection`) to event j, h_j = `bandwidth`,
        phi_j = `p_background`, T = `background.T_days`. Uses the final phi, as the summary
        does (rates()/rates.dat use phi from before the last MLE step: about 1e-7 apart)."""
        from . import __version__
        c, r, phi = self.catalog, self.region, self.final_background_prob
        if self.projection is not None:
            coords = "lonlat"
            ex, ey = self.projection.inverse(c.x, c.y)
            px, py = self.projection.inverse(r.polygon_x, r.polygon_y)
        else:                                       # built without a projection: model x, y
            coords, ex, ey, px, py = "model", c.x, c.y, r.polygon_x, r.polygon_y
        inp = (self.config or {}).get("input", {})
        t0 = getattr(inp.get("catalog"), "t0", None)       # EventCatalog with real times
        params = dataclasses.asdict(self.params)
        if self.K is not None and np.isnan(self.params.A):
            params["K"] = self.K                    # A undefined for p or q <= 1
        checks = self.checks or {"failed": [], "warnings": []}
        b = self.bootstrap
        d = {
            "steqpy_version": __version__,
            "fit": {"valid": not checks["failed"], "failed": checks["failed"],
                    "warnings": checks.get("warnings", []), "converged": self.converged,
                    "neg_loglik": self.neg_loglik, "aic": 2 * (self.neg_loglik + 8),
                    "likelihood_calls": self.n_likelihood_calls,
                    "n_events": len(c), "n_targets": int(np.sum(self.ind == 1)),
                    "usable": checks.get("usable"), "background_share": checks.get("background_share")},
            "params": params,
            "units": {"c": "days", "D": f"deg^2 of arc (1 deg = {DEG_KM} km)",
                      "t_days": "days since t_min (t_min itself in days since t0, when t0 is given)",
                      "bandwidth": "deg of arc (model coordinates)",
                      "coords": "lon/lat in degrees" if coords == "lonlat" else "model x, y"},
            "errors": {
                "status": "experimental: not validated, do not publish yet",
                "hessian": self.hessian_errors,
                "hessian_note": "background held fixed, so too small; "
                                "Wang, Schoenberg & Jackson (2010), BSSA 100(5A), 1989-2001",
                "bootstrap": None if not b or not b["se"] else
                {"se": b["se"], "ci95": b["ci"], "n_good": b["n_good"], "n": len(b["params"]),
                 "reference": "Dutilleul, Genest & Peng (2024), GJI 236, 1601-1608"}},
            "summary": self.summary(),
            "events": {"coords": coords,
                       "x" if coords == "model" else "lon": _sig(ex, 8),
                       "y" if coords == "model" else "lat": _sig(ey, 8),
                       "t_days": _sig(c.t, 10), "t_min": inp.get("t_min"),
                       "t0": None if t0 is None else str(t0),
                       "mag_above_cutoff": _sig(c.m, 4), "mag_cutoff": inp.get("mag_min"),
                       "target": (self.ind == 1).astype(int),
                       "p_background": _sig(phi, 8),
                       "bandwidth": _sig(self.bandwidth, 8)},
            "region": {"polygon": np.column_stack([px, py]), "t_start": r.t_start, "t_end": r.t_end},
            "seeds": {"jitter": inp.get("jitter_seed", 0), "bootstrap": self.settings.bootstrap_seed},
            # half-widths as given (0 = off, "precision"), t in days; the website shows these
            "jitter": {"lon": inp.get("jitter_lon", 0), "lat": inp.get("jitter_lat", 0),
                       "mag": inp.get("jitter_mag", 0), "t_days": inp.get("jitter_t", 0),
                       "seed": inp.get("jitter_seed", 0)},
            "projection": None if self.projection is None else
            {"kind": self.projection.kind, "lon0": self.projection.lon0, "lat0": self.projection.lat0,
             "note": "laea: Lambert azimuthal equal-area at (lon0, lat0), radius 180/pi deg; "
                     "flat: x = lon cos(lat0), y = lat"},
            "background": {"mu": self.params.mu, "T_days": r.t_end - r.t_start,
                           "units": "mu * u in events per day per deg^2 of arc"},
        }
        return _clean(d)

    # -- figures (see plotting; needs matplotlib) -------------------------------------------
    def plot_events(self, kind="all", ax=None, show_outside=True):
        """Map of the events by class: "all", "background", "uncertain" or "aftershock"."""
        from . import plotting
        return plotting.events(self, kind, ax, show_outside)

    def plot_background_rate(self, ax=None, n=250):
        """Map of the background rate (events per year per 10,000 km^2, log scale)."""
        from . import plotting
        return plotting.background_rate(self, ax, n)

    def plot_histogram(self, ax=None):
        """Histogram of P(aftershock) over the target events, coloured by class."""
        from . import plotting
        return plotting.histogram(self, ax)

    def plot_summary(self):
        """One figure: events by class, background rate, histogram, parameters."""
        from . import plotting
        return plotting.summary(self)

    def run_bootstrap(self, n, seed=0, log=None):
        """Parametric bootstrap of this fit (see bootstrap): n catalogues simulated from it, each
        refitted. Costs n fits. Returns a dict: params (n, 8; NaN rows failed), se and ci (95%
        percentile intervals) per parameter, n_good, and the background rate at each event, u,
        with its spread u_se. (Settings.bootstrap = n does the same inside fit().)"""
        from . import bootstrap
        return bootstrap.run(self, n, seed, log or (lambda msg: None))

    def save_json(self, path):
        """Write to_dict() as JSON to path (any name, e.g. "kma2012.json"): the file the
        website opens ("Open a result file"). Returns the Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":")), encoding="utf-8")
        return path

    def write_outputs(self, out):
        """probs.dat, rates.dat (original formats), params.txt, result.json (see to_dict),
        config.toml (resolved: re-running it reproduces this fit)."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "probs.dat", "w") as fh:
            for i in range(len(self.catalog)):
                fh.write(f" {i + 1:8d}{self.background_prob[i]:20.10f}"
                         f"{self.bandwidth[i]:20.10f}{self.background_rate[i]:20.10f}\n")
        self.save_json(out / "result.json")
        if self.grid:
            _, _, rates = self.rate_grid()
            mx, my = self.grid["n"]
            with open(out / "rates.dat", "w") as fh:
                fh.write(f"{mx:12d}\n{my:12d}\n")
                np.savetxt(fh, rates, fmt="%20.10f")
        p = self.params
        lines = [f"{k:6s} = {v:.10g}" for k, v in dataclasses.asdict(p).items()]
        lines[2] += "  # days"
        lines[5] += f"  # deg^2 of arc = {p.D_km2:.10g} km^2"
        if self.K is not None and np.isnan(p.A):
            # A = K / ((p-1)(q-1)) does not exist for p or q <= 1: report the amplitude as K
            lines.append(f"K      = {self.K:.10g}  # A undefined (p or q <= 1); K = A (p-1)(q-1)")
        if self.hessian_errors:
            lines += ["", "# EXPERIMENTAL, do not publish yet: standard errors from the Hessian "
                          "(background held fixed: too small)",
                      "# Wang, Schoenberg & Jackson (2010), BSSA 100(5A), 1989-2001"]
            lines += [f"se({k}){' ' * (6 - len(k))}= {v:.4g}" for k, v in self.hessian_errors.items()]
        if self.bootstrap and self.bootstrap["se"]:
            b = self.bootstrap
            lines += ["", f"# EXPERIMENTAL, do not publish yet: parametric bootstrap standard errors ({b['n_good']} of "
                          f"{len(b['params'])} replicates), 95% percentile intervals",
                      "# Dutilleul, Genest & Peng (2024), GJI 236, 1601-1608 (ETASbootstrap)"]
            lines += [f"se({k}){' ' * (6 - len(k))}= {v:.4g}   95% ({b['ci'][k][0]:.4g}, {b['ci'][k][1]:.4g})"
                      for k, v in b["se"].items()]
        lines += ["", f"-log L = {self.neg_loglik:.10f}",
                  f"AIC    = {2 * (self.neg_loglik + 8):.4f}",
                  f"converged = {self.converged}", f"likelihood calls = "
                  f"{self.n_likelihood_calls}", f"ndiv check = {self.ndiv_error}",
                  f"targets = {int(np.sum(self.ind == 1))} of {len(self.catalog)}"]
        (out / "params.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        if self.bootstrap:
            b = self.bootstrap
            np.savetxt(out / "bootstrap_params.dat", b["params"], fmt="%.10g",
                       header="mu A c alpha p D q gamma  (one replicate per row; NaN = failed)")
            if b["u_se"] is not None:
                np.savetxt(out / "bootstrap_background.dat",
                           np.column_stack([np.arange(1, len(b["u"]) + 1), b["u"], b["u_se"]]),
                           fmt=["%d", "%.10g", "%.10g"],
                           header="event u u_se  (background rate at each event; se over replicates)")
        if self.config:
            c = self.config
            inp = dict(c["input"])
            if not isinstance(inp["catalog"], str):      # in memory (from_catalog): save it
                inp["catalog"].to_etas(out / "catalogue.etas")
                inp["catalog"] = str(out / "catalogue.etas")
            text = config.dump(c["settings"], inp, c["params"], c["grid"], c["preset"], out)
            (out / "config.toml").write_bytes(text.encode("utf-8"))
