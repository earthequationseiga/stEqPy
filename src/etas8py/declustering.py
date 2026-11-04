"""Object-oriented API for stochastic declustering (Zhuang, Ogata & Vere-Jones 2002):
space-time ETAS model with a variable-bandwidth kernel background.

    from etas8py import StochasticDeclustering
    sd = StochasticDeclustering.from_config("configs/jma.toml", threads=8)
    sd.fit()
    sd.params                 # ETASParams (MLE)
    sd.background_prob        # P(event i is a background event), per event
    lon, lat, r = sd.rate_grid()   # background / kernel density / total intensity

Pieces are separate so each can be swapped or used alone: Catalog (events), Region (study
polygon + target window), ETASParams (the 8 model parameters), Settings (algorithm
hyperparameters), StochasticDeclustering (the estimation steps). The numerics live in
`kernels` (Numba). With preset "original" (Settings(converge=False, fast_xint=False,
ndiv_check=False)) this reproduces the original Fortran bit for bit (validated in
tests/test_declustering.py against run.Model).
"""
import dataclasses
import math
import warnings
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

from . import config, geo
from . import kernels as K
from .davidn import davidn

# Original schedule: outer iteration k = 1..11 does (12 - k) background/probability passes,
# then one maximum-likelihood fit. 66 passes in total; there is no convergence test.
FULL_SCHEDULE = tuple(12 - k for k in range(1, 12))
METRICS = ("loglik", "dphi")


@dataclass
class Catalog:
    """Time-sorted events in model coordinates (see geo: x, y in degrees of arc);
    t = days since the catalog start; m = magnitude minus the cutoff (so m >= 0)."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    m: np.ndarray

    def __post_init__(self):
        n = len(self.t)
        if not all(len(a) == n for a in (self.x, self.y, self.m)):
            raise ValueError("x, y, t, m must have the same length")
        if np.any(np.diff(self.t) < 0):
            raise ValueError("events must be sorted by time")

    def __len__(self):
        return len(self.t)


@dataclass
class Region:
    """Study region. polygon_x/_y: vertices in model coordinates (checked, closed and made
    counter-clockwise on construction). Events inside the polygon with
    t_start <= t <= t_end are the *targets* whose likelihood is fitted; earlier/outside
    events still trigger (they enter the intensity)."""
    polygon_x: np.ndarray
    polygon_y: np.ndarray
    t_start: float
    t_end: float

    def __post_init__(self):
        self.polygon_x, self.polygon_y = geo.check_planar_polygon(
            self.polygon_x, self.polygon_y, lambda m: warnings.warn(m.strip()))
        if not self.t_start < self.t_end:
            raise ValueError("need t_start < t_end")

    @property
    def npoly(self):
        return len(self.polygon_x) - 1

    def target_flags(self, cat):
        """1 = target, 0 = on the boundary / outside the window, -1 = outside the polygon
        (the original's `ind` convention; only 1 counts)."""
        from .run import polyse
        ind = polyse(self.polygon_x, self.polygon_y, cat.x, cat.y)
        ind[(cat.t > self.t_end) | (cat.t < self.t_start)] = 0
        return ind


@dataclass
class ETASParams:
    """Space-time ETAS parameters. Conditional intensity
        lambda(t,x,y) = mu u(x,y) + sum_{t_j<t} A e^{alpha m_j} (p-1)/c (1+dt/c)^-p
                         * (q-1)/(pi D e^{gamma m_j}) (1 + r^2/(D e^{gamma m_j}))^-q
    mu: background scale; A, alpha: productivity; c (days), p: Omori-Utsu time decay;
    D (deg^2 of arc), q, gamma: spatial spread (magnitude-scaled) and its decay."""
    mu: float
    A: float
    c: float
    alpha: float
    p: float
    D: float
    q: float
    gamma: float

    def to_array(self):
        return np.array(astuple(self), dtype=np.float64)

    @classmethod
    def from_array(cls, a):
        return cls(*map(float, a))

    @property
    def D_km2(self):
        return self.D * geo.KM_PER_DEG ** 2

    def __post_init__(self):
        if any(v <= 0 for v in astuple(self)):
            raise ValueError("all ETAS parameters must be positive")


@dataclass(frozen=True)
class Settings:
    """Algorithm hyperparameters (TOML tables in config.SECTIONS). Defaults = preset
    "default"; preset "original" = converge=False, fast_xint=False, ndiv_check=False."""
    # [model]
    projection: str = "flat"        # "flat" (original) or "laea" (equal-area), see geo
    npp: int = 5                    # background bandwidth = distance to npp-th nearest event
    bwm: str = "0.05 deg"           # ...floored at bwm; units required ("deg" or "km")
    optimise: bool = True           # False: keep the initial ETAS parameters fixed
    # [convergence]
    converge: bool = True           # False: run the fixed `schedule` (the original)
    schedule: tuple = FULL_SCHEDULE  # bkgd/prob passes per outer iteration (each + 1 MLE)
    outer_metric: str = "loglik"    # loglik: rel. change of -log L; dphi: max |d phi_i|
    outer_tol: float = 1e-8         # JMA: stops after 5 of 11 outer iterations
    max_outer: int = 11
    inner_metric: str = "dphi"      # per background/probability pass
    inner_tol: float = 1e-4
    max_inner: int = 11
    # [run]
    threads: int = 0                # 0 = serial loops; >0 = Numba threads (bit-identical)
    # [advanced]
    ndiv: int = 1000                # pieces per polygon edge in the spatial integrals
    ndiv_check: bool = True         # one-off: -log L change when ndiv is doubled
    ndiv_check_tol: float = 1e-6    # ...warn above this relative change
    fast_xint: bool = True          # fused polygon integrals (same result to ~1e-12)
    tol: float = 1e-6               # optimiser stopping tolerance (gradient/step/rel. f)
    max_sweeps: int = 10            # optimiser: max sweeps of 8 quasi-Newton steps
    ls_step: float = 0.05           # line search: first step, bracketing factors
    ls_shrink: float = 0.1
    ls_expand: float = 2.0
    frint_eps: float = K.E10        # polygon-integral guards (see kernels.frint)
    frint_eps_r: float = K.E20
    log_floor: float = 1e-25        # a target with lambda <= log_floor scores -log_penalty
    log_penalty: float = 100.0
    chunksize: int = 16             # Numba chunk size (balances xlamb's O(i) cost)

    def __post_init__(self):
        geo.length_deg(self.bwm, "bwm")                 # units check
        if self.projection not in ("flat", "laea"):
            raise ValueError("projection must be 'flat' or 'laea'")
        if self.inner_metric not in METRICS or self.outer_metric not in METRICS:
            raise ValueError(f"convergence metrics must be one of {METRICS}")
        if (self.npp < 1 or self.ndiv < 1 or self.max_sweeps < 1 or self.max_outer < 1
                or self.max_inner < 1 or self.chunksize < 1):
            raise ValueError("need npp, ndiv, max_sweeps, max_outer, max_inner, "
                             "chunksize >= 1")
        if min(self.tol, self.outer_tol, self.inner_tol, self.ndiv_check_tol, self.ls_step,
               self.frint_eps, self.frint_eps_r, self.log_floor) <= 0:
            raise ValueError("tolerances, guards and ls_step must be > 0")
        if not (0 < self.ls_shrink < 1 < self.ls_expand):
            raise ValueError("need 0 < ls_shrink < 1 < ls_expand")
        if not self.schedule or not all(int(k) >= 0 for k in self.schedule):
            raise ValueError("schedule must be a non-empty list of passes >= 0")
        import numba
        if not 0 <= self.threads <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f"threads must be 0..{numba.config.NUMBA_NUM_THREADS} "
                             "(NUMBA_NUM_THREADS)")
        if (self.converge and self.fast_xint and "loglik" in
                (self.inner_metric, self.outer_metric) and min(
                    self.inner_tol if self.inner_metric == "loglik" else 1,
                    self.outer_tol if self.outer_metric == "loglik" else 1) < 1e-9):
            warnings.warn("loglik tolerance < 1e-9 is below the fast_xint rounding level "
                          "(~3e-10): it may never be reached")

    @property
    def bwm_deg(self):
        return geo.length_deg(self.bwm, "bwm")


class StochasticDeclustering:
    """Alternates (a) background estimation given the background probabilities and
    (b) ETAS maximum likelihood given the background, then (c) probabilities given both.
    State after fit(): params, background_prob, background_rate, bandwidth, history,
    converged, neg_loglik, ndiv_error."""

    def __init__(self, catalog, region, init_params, settings=Settings(), log=None,
                 projection=None, grid=None):
        self.catalog = catalog
        self.region = region
        self.settings = settings
        self.params = init_params
        self.log = log or (lambda msg: None)
        self.projection = projection            # geo.Projection, for lon/lat outputs
        self.grid = grid                        # {"lon": [a, b], "lat": [a, b], "n": [.,.]}
        self.config = None                      # resolved config (set by from_config)
        self.ind = region.target_flags(catalog)
        n = len(catalog)
        if not np.any(self.ind == 1):
            raise ValueError("no target events (inside the polygon and the time window)")
        if not np.any(catalog.t < region.t_start):
            warnings.warn("no events before t_start: early targets miss their triggers")
        if np.all(self.ind != -1):
            warnings.warn("no events outside the polygon: targets near the boundary miss "
                          "triggers from outside")
        self.background_prob = np.ones(n)       # zprob: start as "everything is background"
        self.background_rate = np.zeros(n)      # zbkgd: u(x_i, y_i) per unit time
        self.bandwidth = None                   # zbandw
        self.xint0 = 0.0                        # integral of the background over the region
        self.n_likelihood_calls = 0
        self.neg_loglik = None                  # -log L at params (after maximise)
        self.history = []                       # (level, outer, pass, change)
        self.converged = None
        self.ndiv_error = None

    @classmethod
    def from_config(cls, source=None, log=None, **overrides):
        """Build from a TOML file / dict (see config). overrides: Settings fields or
        preset=, applied last."""
        from .run import read_catalog
        cfg = config.load(source, **overrides)
        s = Settings(**cfg["settings"])
        inp = cfg["input"]
        poly = inp["polygon"]
        plon, plat = (geo.read_polygon(poly) if isinstance(poly, str)
                      else np.array(poly, dtype=np.float64).T)
        plon, plat = geo.check_lonlat_polygon(plon, plat, lambda m: warnings.warn(m.strip()))
        proj = geo.Projection(s.projection, plon, plat)
        # (_t_span: a control file's interval length, kept exact for bit-reproduction)
        t_span = inp.get("_t_span", inp["t_end"] - inp["t_min"])
        lon, lat, mag, t = read_catalog(inp["catalog"], inp["mag_min"], inp["t_min"], t_span)
        x, y = proj(lon, lat)
        px, py = proj(plon, plat)
        cat = Catalog(x, y, t - inp["t_min"], mag - inp["mag_min"])
        reg = Region(np.append(px, px[0]), np.append(py, py[0]),
                     inp["t_start"] - inp["t_min"], t_span)
        p = dict(cfg["params"])
        p["D"] = geo.area_deg2(p["D"], "params.D")
        sd = cls(cat, reg, ETASParams(**{k: float(v) for k, v in p.items()}), s, log, proj,
                 cfg["grid"] or None)
        cfg["settings"] = dataclasses.asdict(s)
        sd.config = cfg
        return sd

    @classmethod
    def from_input_file(cls, path, log=None, **overrides):
        """Build from an original-format control file (e.g. jap.in); same as a config
        with only [input] control = path."""
        return cls.from_config({"input": {"control": str(path)}}, log, **overrides)

    # -- the estimation steps (each usable on its own) ----------------------------------
    def _k(self, name):
        return getattr(K, name + "_par") if self.settings.threads > 0 else getattr(K, name)

    def _use_threads(self):
        # process-global: the last fit()/rates() call decides for every instance
        if self.settings.threads > 0:
            import numba
            numba.set_num_threads(self.settings.threads)
            numba.set_parallel_chunksize(self.settings.chunksize)

    def compute_bandwidths(self):
        c, s = self.catalog, self.settings
        self.bandwidth = K.bandwcalc(c.x, c.y, s.npp, s.bwm_deg)

    def _bkgd(self, out, ndiv):
        c, r, s = self.catalog, self.region, self.settings
        return self._k("bkgdcalc")(c.x, c.y, self.background_prob, self.bandwidth,
                                   r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start,
                                   out, ndiv, s.frint_eps, s.frint_eps_r)

    def update_background(self):
        """u(x_i, y_i) = sum_j phi_j g_{h_j}(r_ij) / T and its integral over the region."""
        self.xint0 = self._bkgd(self.background_rate, self.settings.ndiv)

    def update_probabilities(self):
        """phi_i = mu u_i / lambda_i."""
        c = self.catalog
        self._k("probcalc")(np.sqrt(self.params.to_array()), self.background_rate, c.x, c.y,
                            c.t, c.m, self.background_prob)

    def _func15(self, sqrt_params, xint0, ndiv):
        c, r, s = self.catalog, self.region, self.settings
        grad = np.empty(8)
        f = self._k("func15")(np.asarray(sqrt_params, dtype=np.float64), self.ind,
                              self.background_rate, xint0, c.x, c.y, c.t, c.m,
                              r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start, grad,
                              s.fast_xint, ndiv, s.frint_eps, s.frint_eps_r, s.log_floor,
                              s.log_penalty)
        return f, list(grad)

    def neg_log_likelihood(self, sqrt_params):
        """(-log L, gradient w.r.t. sqrt(params)) with the background held fixed."""
        self.n_likelihood_calls += 1
        return self._func15(sqrt_params, self.xint0, self.settings.ndiv)

    def maximise(self):
        """Quasi-Newton (DFP) MLE on sqrt(params), which keeps every parameter positive."""
        s = self.settings
        x = list(np.sqrt(self.params.to_array()))
        last = []

        def f(b, log):
            r = self.neg_log_likelihood(b)
            last[:] = [r[0]]
            return r
        davidn(x, f, self.log, s.tol, s.max_sweeps, s.ls_step, s.ls_shrink, s.ls_expand)
        self.params = ETASParams.from_array(np.array(x) ** 2)
        self.neg_loglik = last[0]               # davidn's last call is at the returned x

    def check_ndiv(self):
        """Relative change of -log L (current params and background) when the polygon
        edges are cut into 2*ndiv pieces instead of ndiv. Large = quadrature too coarse
        for the narrowest kernels near the boundary."""
        s = self.settings
        sq = np.sqrt(self.params.to_array())
        x2 = self._bkgd(np.empty(len(self.catalog)), 2 * s.ndiv)
        f1 = self._func15(sq, self.xint0, s.ndiv)[0]
        f2 = self._func15(sq, x2, 2 * s.ndiv)[0]
        self.ndiv_error = abs(f2 - f1) / abs(f2)
        self.log(f" ndiv check: -log L {f1:.10f} (ndiv {s.ndiv}) vs {f2:.10f} "
                 f"(ndiv {2 * s.ndiv}): relative change {self.ndiv_error:.2e}")
        if self.ndiv_error > s.ndiv_check_tol:
            warnings.warn(f"ndiv={s.ndiv} changes -log L by {self.ndiv_error:.1e} when "
                          f"doubled (> {s.ndiv_check_tol}): consider a larger ndiv")

    def _change(self, metric, ref):
        """(change, new reference) for a convergence test; ref None = first evaluation.
        dphi: max |phi - ref| (ref = earlier phi array); loglik: relative change of
        -log L at the current params and background (ref = earlier -log L)."""
        if metric == "dphi":
            ch = np.inf if ref is None else float(np.max(np.abs(self.background_prob - ref)))
            return ch, self.background_prob.copy()
        f = self._func15(np.sqrt(self.params.to_array()), self.xint0, self.settings.ndiv)[0]
        return (np.inf if ref is None else abs(f - ref) / abs(f)), f

    def fit(self):
        """Fixed schedule (converge=False, the original) or: outer iterations until the
        outer metric changes by < outer_tol (max max_outer), each running background/
        probability passes until the inner metric changes by < inner_tol (max max_inner)
        and then the MLE."""
        s = self.settings
        self._use_threads()
        self.compute_bandwidths()
        self.history = []
        n_outer = s.max_outer if s.converge else len(s.schedule)
        ref_outer = None
        self.converged = False if s.converge else None
        for k in range(1, n_outer + 1):
            self.log(f" Iteration {k}")
            n_inner = s.max_inner if s.converge else s.schedule[k - 1]
            ref = None
            for r in range(1, n_inner + 1):
                if s.converge and s.inner_metric == "dphi":
                    ref = self.background_prob.copy()    # change within this pass
                self.update_background()
                if s.ndiv_check and self.ndiv_error is None:
                    self.check_ndiv()
                self.update_probabilities()
                if s.converge:
                    ch, ref = self._change(s.inner_metric, ref)
                    self.history.append(("inner", k, r, ch))
                    self.log(f"  pass {r}: {s.inner_metric} change {ch:.3e}")
                    if ch < s.inner_tol:
                        break
            if s.optimise:
                self.maximise()
                self.log(f" mle = {self.params}")
            if s.converge:
                ch, ref_outer = self._change(s.outer_metric, ref_outer)
                self.history.append(("outer", k, 0, ch))
                self.log(f" outer {k}: {s.outer_metric} change {ch:.3e}")
                if ch < s.outer_tol:
                    self.converged = True
                    break
        if s.converge and not self.converged:
            warnings.warn(f"not converged after max_outer={s.max_outer} iterations")
        if self.neg_loglik is None:
            self.neg_loglik = self._func15(np.sqrt(self.params.to_array()), self.xint0,
                                           s.ndiv)[0]
        return self

    # -- outputs ------------------------------------------------------------------------
    def rates(self, gx, gy):
        """(len(gx)*len(gy), 3): background rate, unweighted kernel density, total
        intensity at t_end, x-major. gx, gy in model coordinates."""
        c, r = self.catalog, self.region
        self._use_threads()
        return self._k("outrates_grid")(np.sqrt(self.params.to_array()), c.x, c.y, c.t, c.m,
                                        self.background_prob, self.bandwidth, r.t_end,
                                        r.t_start, np.asarray(gx, float),
                                        np.asarray(gy, float))

    def rates_lonlat(self, lon, lat):
        """rates() at arbitrary lon/lat points (1-D arrays of equal length)."""
        c, r = self.catalog, self.region
        px, py = self.projection(lon, lat)
        self._use_threads()
        return self._k("outrates_points")(np.sqrt(self.params.to_array()), c.x, c.y, c.t,
                                          c.m, self.background_prob, self.bandwidth, r.t_end,
                                          r.t_start, px, py)

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

    def write_outputs(self, out):
        """probs.dat, rates.dat (original formats), params.txt, config.toml (resolved:
        re-running it reproduces this fit)."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "probs.dat", "w") as fh:
            for i in range(len(self.catalog)):
                fh.write(f" {i + 1:8d}{self.background_prob[i]:20.10f}"
                         f"{self.bandwidth[i]:20.10f}{self.background_rate[i]:20.10f}\n")
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
        lines += ["", f"-log L = {self.neg_loglik:.10f}",
                  f"AIC    = {2 * (self.neg_loglik + 8):.4f}",
                  f"converged = {self.converged}", f"likelihood calls = "
                  f"{self.n_likelihood_calls}", f"ndiv check = {self.ndiv_error}",
                  f"targets = {int(np.sum(self.ind == 1))} of {len(self.catalog)}"]
        (out / "params.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        if self.config:
            c = self.config
            text = config.dump(c["settings"], c["input"], c["params"], c["grid"],
                               c["preset"])
            (out / "config.toml").write_bytes(text.encode("utf-8"))


__all__ = ["Catalog", "Region", "ETASParams", "Settings", "StochasticDeclustering",
           "FULL_SCHEDULE"]
