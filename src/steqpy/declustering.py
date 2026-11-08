"""The estimator for stochastic declustering: the method of Zhuang, Ogata & Vere-Jones (2002)
as implemented in Zhuang's etas8p (space-time ETAS with a variable-bandwidth kernel
background; the model details follow that code).

    from steqpy import EventCatalog, StochasticDeclustering
    sd = StochasticDeclustering.from_config("configs/jma.toml", threads=8)
    # or from data in memory (see events.EventCatalog):
    # events = EventCatalog.from_csv("cat.csv")
    # sd = StochasticDeclustering.from_catalog(events, polygon, "2000-01-01", "2024-12-31", 5.2)
    result = sd.fit()
    result.params                 # ETASParams (MLE)
    result.background_prob        # P(event i is a background event), per event
    lon, lat, r = result.rate_grid()   # background / kernel density / total intensity
    result.sample_background(seed=1)   # one declustered catalogue, as a boolean mask
    result.sample_parents(seed=1)      # one family tree: parent index, -1 = background
    result.triggering_probs(i)         # P(each earlier event triggered event i)

Pieces are separate so each can be swapped or used alone: Catalog (events), Region (study
polygon + target window), ETASParams (the 8 model parameters), Settings (algorithm
hyperparameters), StochasticDeclustering (the estimation steps), DeclusteringResult (what
a fit produces). The numerics live in `kernels` (Numba). With preset "original"
(Settings(converge=False, quadrature="ndiv", fast_xint=False, ndiv_check=False)) this
reproduces the original Fortran bit for bit (validated in tests/test_declustering.py against
run.Model). Default results differ from it: mainly the equal-area projection (JMA: parameters
by up to 1.5%, phi by up to 0.09), and slightly the more accurate polygon integrals (3e-6).
The default also fits with BFGS (see bfgs) in the K form (see kernels, K = A (p-1)(q-1)): same
MLE when p, q > 1, about 5x fewer likelihood calls, and no runaway when a step crosses 1.
"""
import dataclasses
import warnings

import numpy as np

from . import config, geo
from . import kernels as K
from . import bootstrap
from .bfgs import bfgs
from .catalog import Catalog, Region
from .davidn import davidn
from .events import EventCatalog
from .params import ETASParams, Settings
from .readers import read_catalog, select_rows
from .result import DeclusteringResult


# check_fit limits. At an optimum in (mu, A) the integral of lambda equals the number of targets,
# and the valid JMA fits have |g| of 5e-3 or less, so these only catch fits that are far off.
INTEGRAL_TOL = 0.01
GRADIENT_TOL = 1.0
# Plausibility (warnings only, the fit still counts): an optimum can be real and still say the
# kernel shape does not suit the data, e.g. q -> infinity is the Gaussian limit of the spatial
# kernel. Bounds are generous: published ETAS fits sit well inside them. c in days, alpha and
# gamma per magnitude unit.
PLAUSIBLE = {"c": (1e-6, 10.0), "alpha": (0.0, 5.0), "p": (0.0, 3.0), "q": (0.0, 10.0),
             "gamma": (0.0, 5.0)}
NAMES = ("mu", "A", "c", "alpha", "p", "D", "q", "gamma")
# Below this expected share of background among the targets, warn: nearly everything is read as
# triggered (the incomplete KMA and Indonesia windows gave 0-5%; valid fits here gave 11-46%).
BG_SHARE_MIN = 0.1
EXPERIMENTAL = ("experimental, not for publication yet: {} (see status.md). The point estimates "
                "and probabilities are unaffected.")
# Start values when none are given (from_catalog): the JMA fit (configs/jma.toml). The fit
# starts from these and moves to the MLE; they only need p, q > 1 and the right scale.
DEFAULT_START = {"mu": 0.59284459, "A": 0.204288231, "c": 0.022692883, "alpha": 1.495169224,
                 "p": 1.109752319, "D": "0.001175925 deg2", "q": 1.86004421, "gamma": 1.041549634}


class StochasticDeclustering:
    """Alternates (a) background estimation given the background probabilities and
    (b) ETAS maximum likelihood given the background, then (c) probabilities given both.
    fit() returns a DeclusteringResult. The attributes below hold the working state, for
    using the steps one at a time; result() snapshots it."""

    def __init__(self, catalog, region, init_params, settings=Settings(), log=None,
                 projection=None, grid=None):
        self.catalog = catalog
        self.region = region
        self.settings = settings
        self.init_params = init_params
        self.log = log or (lambda msg: None)
        self.projection = projection            # geo.Projection, for lon/lat outputs
        self.grid = grid                        # {"lon": [a, b], "lat": [a, b], "n": [.,.]}
        self.config = None                      # resolved config (set by from_config)
        self.ind = region.target_flags(catalog)
        if not np.any(self.ind == 1):
            raise ValueError("no target events (inside the polygon and the time window)")
        if not np.any(catalog.t < region.t_start):
            warnings.warn("no events before t_start: early targets miss their triggers")
        if np.all(self.ind != -1):
            warnings.warn("no events outside the polygon: targets near the boundary miss "
                          "triggers from outside")
        self._reset()

    def _reset(self):
        n = len(self.catalog)
        self.params = self.init_params
        self.background_prob = np.ones(n)       # zprob: start as "everything is background"
        self.background_rate = np.zeros(n)      # zbkgd: u(x_i, y_i) per unit time
        self.bandwidth = None                   # zbandw
        self.xint0 = 0.0                        # integral of the background over the region
        self.n_likelihood_calls = 0
        self.neg_loglik = None                  # -log L at params (after maximise)
        self.history = []                       # (level, outer, pass, change)
        self.converged = None
        self.ndiv_error = None
        self.checks = None
        self.K = None                           # K form: K, kept for when A is undefined
        self.hessian_errors = None              # {name: se}, Settings.hessian_errors
        self.bootstrap = None                   # bootstrap.run's dict, Settings.bootstrap
        self.bfgs_H = None                      # BFGS inverse Hessian, kept between outers

    @classmethod
    def from_config(cls, source=None, log=None, **overrides):
        """Build from a TOML file / dict (see config). overrides: Settings fields or
        preset=, applied last."""
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
        jitter = {k: inp.get(f"jitter_{k}", 0.0) for k in ("lon", "lat", "mag", "t")}
        seed = int(inp.get("jitter_seed", 0))
        if isinstance(inp["catalog"], EventCatalog):     # in memory (from_catalog)
            lon, lat, mag, t = select_rows(inp["catalog"].rows(), inp["mag_min"], inp["t_min"],
                                           t_span, jitter, seed, "EventCatalog")
        else:
            lon, lat, mag, t = read_catalog(inp["catalog"], inp["mag_min"], inp["t_min"], t_span,
                                            jitter, seed)
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
    def from_catalog(cls, events, polygon, t_start, t_end, mag_min, params=None, t_min=None,
                     jitter=None, seed=0, grid=None, log=None, **overrides):
        """Build from an EventCatalog in memory. polygon: [[lon, lat], ...] or a polygon file;
        t_start, t_end: the target window (dates, or days since events.t0); mag_min: the
        magnitude cutoff (see events.completeness); params: start values {mu, A, ..., D: "...
        deg2"} (default DEFAULT_START); t_min: earliest time read (default one day before the
        first event: every event can trigger); jitter: {lon, lat, mag, t: half-width or
        "precision"} (see readers.jitter_rows); seed: for every random step (jitter, bootstrap;
        Settings.bootstrap_seed in overrides wins for the bootstrap); grid: rate-map grid {lon, lat, n};
        overrides: Settings fields or preset=. Same checks and defaults as from_config."""
        inp = {"catalog": events,
               "polygon": polygon if isinstance(polygon, str) else [[float(a), float(b)] for a, b in polygon],
               "t_min": float(events.t.min()) - 1.0 if t_min is None else events.days(t_min),
               "t_start": events.days(t_start), "t_end": events.days(t_end), "mag_min": float(mag_min)}
        if jitter:
            inp.update({f"jitter_{k}": v for k, v in jitter.items()}, jitter_seed=int(seed))
        overrides.setdefault("bootstrap_seed", int(seed))
        source = {"input": inp, "params": dict(DEFAULT_START if params is None else params)}
        if grid:
            source["grid"] = grid
        return cls.from_config(source, log, **overrides)

    @classmethod
    def from_input_file(cls, path, log=None, **overrides):
        """Build from an original-format control file (e.g. jap.in); same as a config
        with only [input] control = path."""
        return cls.from_config({"input": {"control": str(path)}}, log, **overrides)

    # -- the estimation steps (each usable on its own) ----------------------------------
    def compute_bandwidths(self):
        c, s = self.catalog, self.settings
        # same bits; "ndiv" and "original" keep the 1:1 search
        search = K.bandwcalc_tree if s.quadrature == "sgl" else K.bandwcalc
        self.bandwidth = search(c.x, c.y, s.npp, s.bwm_deg)

    def _bkgd(self, out, refine=1):
        """refine multiplies the quadrature size, for check_ndiv."""
        c, r, s = self.catalog, self.region, self.settings
        args = (c.x, c.y, self.background_prob, self.bandwidth, r.polygon_x, r.polygon_y,
                r.npoly, r.t_end, r.t_start, out)
        if s.quadrature == "sgl":
            return s.kernel("bkgdcalc_sgl")(*args, *K.gauss_nodes(refine * s.sgl_nodes_background))
        return s.kernel("bkgdcalc")(*args, refine * s.ndiv, s.frint_eps, s.frint_eps_r)

    def update_background(self):
        """u(x_i, y_i) = sum_j phi_j g_{h_j}(r_ij) / T and its integral over the region."""
        self.xint0 = self._bkgd(self.background_rate)

    def update_probabilities(self):
        """phi_i = mu u_i / lambda_i."""
        c = self.catalog
        if self.settings.form == "K":
            b = np.sqrt(self._theta())
            lam = np.empty(len(c))
            self.settings.kernel("lambda_k")(b, self.background_rate, c.x, c.y, c.t, c.m, lam)
            self.background_prob[:] = b[0] ** 2 * self.background_rate / lam
            return
        self.settings.probcalc()(np.sqrt(self.params.to_array()), self.background_rate, c.x, c.y,
                                 c.t, c.m, self.background_prob)

    def _theta(self):
        """The optimiser's parameters: params, with K in A's place in the K form."""
        th = self.params.to_array().copy()
        if self.settings.form == "K":
            th[1] = self.K if np.isnan(th[1]) else th[1] * (th[4] - 1) * (th[6] - 1)
        return th

    def _set_theta(self, th):
        """Inverse of _theta. In the K form A = K / ((p-1)(q-1)) exists only for p, q > 1;
        otherwise A is NaN and K holds the fit."""
        th = np.array(th, dtype=float)
        if self.settings.form == "K":
            self.K = th[1]
            th[1] = th[1] / ((th[4] - 1) * (th[6] - 1)) if th[4] > 1 and th[6] > 1 else np.nan
        self.params = ETASParams.from_array(th)

    def _func15_k(self, sqrt_theta, xint0, refine=1):
        """K form: (-log L, gradient, lambda; lambda is valid at targets only)."""
        c, r, s = self.catalog, self.region, self.settings
        grad = np.empty(8)
        f, lam = s.kernel("func15_k")(
            np.asarray(sqrt_theta, dtype=np.float64), self.ind, self.background_rate, xint0,
            c.x, c.y, c.t, c.m, r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start, grad,
            *K.gauss_nodes(refine * s.sgl_nodes), s.log_floor, s.log_penalty)
        return f, list(grad), lam

    def _func15(self, sqrt_params, xint0, refine=1):
        """(-log L, gradient) at sqrt(_theta()) values, in the configured form."""
        c, r, s = self.catalog, self.region, self.settings
        if s.form == "K":
            return self._func15_k(sqrt_params, xint0, refine)[:2]
        grad = np.empty(8)
        args = (np.asarray(sqrt_params, dtype=np.float64), self.ind, self.background_rate, xint0,
                c.x, c.y, c.t, c.m, r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start, grad)
        if s.quadrature == "sgl":
            f = s.kernel("func15_sgl")(*args, *K.gauss_nodes(refine * s.sgl_nodes), s.log_floor,
                                       s.log_penalty, s.xlamb == "logs")
        else:
            f = s.kernel("func15")(*args, s.fast_xint, refine * s.ndiv, s.frint_eps,
                                   s.frint_eps_r, s.log_floor, s.log_penalty)
        return f, list(grad)

    def neg_log_likelihood(self, sqrt_params):
        """(-log L, gradient w.r.t. sqrt(params)) with the background held fixed."""
        self.n_likelihood_calls += 1
        return self._func15(sqrt_params, self.xint0)

    def _value(self, sqrt_params, log):
        """-log L alone, same bits as neg_log_likelihood's, for the line search."""
        c, r, s = self.catalog, self.region, self.settings
        self.n_likelihood_calls += 1
        return s.kernel("func15_sgl_value")(
            np.asarray(sqrt_params, dtype=np.float64), self.ind, self.background_rate, self.xint0,
            c.x, c.y, c.t, c.m, r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start,
            *K.gauss_nodes(s.sgl_nodes), s.log_floor, s.log_penalty, s.xlamb == "logs")

    def maximise(self):
        """Quasi-Newton MLE on sqrt(theta), which keeps every parameter positive: BFGS
        (default) or davidn (DFP, the original's)."""
        s = self.settings
        x = list(np.sqrt(self._theta()))
        last = []

        def f(b, log):
            r = self.neg_log_likelihood(b)
            last[:] = [r[0]]
            return r
        if s.optimiser == "bfgs":
            self.bfgs_H = bfgs(x, f, self.log, self.bfgs_H, s.tol, s.max_sweeps, s.ls_step)
        else:
            davidn(x, f, self.log, s.tol, s.max_sweeps, s.ls_step, s.ls_shrink, s.ls_expand,
                   value=self._value if s.quadrature == "sgl" and s.form == "A" else None)
        self._set_theta(np.array(x) ** 2)
        self.neg_loglik = last[0]               # the last call is at the returned x

    def check_ndiv(self):
        """Relative change of -log L (current params and background) when the polygon
        quadrature size is doubled (sgl nodes, or ndiv pieces). Large = quadrature too
        coarse for the narrowest kernels near the boundary."""
        s = self.settings
        sq = np.sqrt(self._theta())
        x2 = self._bkgd(np.empty(len(self.catalog)), 2)
        f1 = self._func15(sq, self.xint0)[0]
        f2 = self._func15(sq, x2, 2)[0]
        self.ndiv_error = abs(f2 - f1) / abs(f2)
        name = "sgl_nodes" if s.quadrature == "sgl" else "ndiv"
        size = s.sgl_nodes if s.quadrature == "sgl" else s.ndiv
        self.log(f" {name} check: -log L {f1:.10f} ({name} {size}) vs {f2:.10f} "
                 f"({name} {2 * size}): relative change {self.ndiv_error:.2e}")
        if self.ndiv_error > s.ndiv_check_tol:
            warnings.warn(f"{name}={size} changes -log L by {self.ndiv_error:.1e} when "
                          f"doubled (> {s.ndiv_check_tol}): consider a larger {name}")

    def standard_errors(self, rel_step=1e-4):
        """Approximate standard errors of the reported parameters: inverse of the observed
        information (Hessian of -log L, central differences of the analytic gradient in
        sqrt(theta)), mapped to the parameters by the delta method. Conditional on the
        background, which the likelihood holds fixed (a kernel smoother of phi, not a fitted
        parameter), so somewhat too small; the bootstrap includes it. In the K form
        A = K/((p-1)(q-1)) gets its error through K, p and q; K's is given when A is undefined.
        NaN if the Hessian is not positive definite (not a clear minimum).
        Reference: Wang, Q., Schoenberg, F. P. & Jackson, D. D. (2010). Standard errors of
        parameter estimates in the ETAS model. BSSA 100(5A), 1989-2001, doi:10.1785/0120100001
        (limits of Hessian errors for ETAS; prefer Settings.bootstrap where the cost allows)."""
        warnings.warn(EXPERIMENTAL.format("Hessian standard errors hold the background fixed, so "
                                          "they are too small"))
        th = self._theta()
        b = np.sqrt(th)
        H = np.empty((8, 8))
        for k in range(8):
            e = rel_step * b[k]
            bp, bm = b.copy(), b.copy()
            bp[k] += e
            bm[k] -= e
            H[:, k] = (np.array(self._func15(bp, self.xint0)[1])
                       - np.array(self._func15(bm, self.xint0)[1])) / (2 * e)
        H = 0.5 * (H + H.T)
        names = list(NAMES)
        J = np.diag(2.0 * b)                    # d theta_k / d b_k
        if self.settings.form == "K":
            p, q = th[4], th[6]
            if p > 1 and q > 1:                 # A = b1^2 / ((b4^2 - 1)(b6^2 - 1))
                a = th[1] / ((p - 1) * (q - 1))
                J[1, 1] = 2.0 * b[1] / ((p - 1) * (q - 1))
                J[1, 4] = -a * 2.0 * b[4] / (p - 1)
                J[1, 6] = -a * 2.0 * b[6] / (q - 1)
            else:
                names[1] = "K"
        with np.errstate(all="ignore"):
            if np.all(np.isfinite(H)) and np.linalg.eigvalsh(H).min() > 0:
                se = np.sqrt(np.diag(J @ np.linalg.inv(H) @ J.T))
            else:
                se = np.full(8, np.nan)
        return dict(zip(names, (float(v) for v in se)))

    def check_fit(self):
        """Is the fit a usable model? -log L settling (the outer test) does not show this: outside
        p, q > 1 it has no lower limit, so an invalid fit can settle or keep "improving". Fills
        self.checks; a failure warns and makes converged False."""
        c, s = self.catalog, self.settings
        th = self._theta()
        t = self.ind == 1
        with np.errstate(all="ignore"):
            b = np.sqrt(th)
            if s.form == "K":
                f, g, lam = self._func15_k(b, self.xint0)
                lam = lam[t]
            else:
                f, g = self._func15(b, self.xint0)
                phi = np.empty(len(c))
                s.probcalc()(b, self.background_rate, c.x, c.y, c.t, c.m, phi)
                lam = th[0] * self.background_rate[t] / phi[t]
            logs = np.where(lam > s.log_floor, np.log(lam), -s.log_penalty)
        integral = f + float(logs.sum())
        ratio = integral / int(t.sum())
        gnorm = float(np.linalg.norm(g))
        self.checks = dict(p=float(th[4]), q=float(th[6]), n_lambda_not_positive=int(np.sum(~(lam > 0))),
                           integral=integral, integral_over_n=ratio, gradient_norm=gnorm)
        failed = []
        if not (np.all(np.isfinite(th)) and np.isfinite(f)):
            failed.append("parameters or -log L not finite")
        # The A form is normalised over infinite time and space, which needs both above 1.
        # The K form is valid for any p, q (A is then undefined and only K is reported).
        if s.form == "A" and not (th[4] > 1 and th[6] > 1):
            failed.append(f"p = {th[4]:.4g}, q = {th[6]:.4g}: the kernel needs both above 1")
        if self.checks["n_lambda_not_positive"]:
            failed.append(f"{self.checks['n_lambda_not_positive']} targets have lambda <= 0")
        if not integral > 0:
            failed.append(f"integral of lambda = {integral:.4g}")
        if s.optimise:      # with fixed parameters the fit is not at an optimum, so these need not hold
            if not abs(ratio - 1) <= INTEGRAL_TOL:
                failed.append(f"integral of lambda / targets = {ratio:.4g}, should be 1")
            if not gnorm <= GRADIENT_TOL:
                failed.append(f"gradient norm = {gnorm:.4g}")
        self.checks["failed"] = failed
        # Plausibility: warnings only. The fit is valid but may not describe the data well.
        notes = []
        vals = self.params.to_array()
        for i, name in enumerate(NAMES):
            if name in PLAUSIBLE:
                lo, hi = PLAUSIBLE[name]
                if not lo < vals[i] < hi:
                    notes.append(f"{name} = {vals[i]:.4g} is outside the usual range ({lo:g}, {hi:g})")
        # Signs the fit does not describe the data (the website refuses p or q <= 1). Usually the
        # catalogue is incomplete for part of the window, which reads as clustering.
        if s.form == "K" and not (th[4] > 1 and th[6] > 1):
            notes.append(f"p = {th[4]:.4g}, q = {th[6]:.4g}: one is <= 1, so A is undefined and "
                         "aftershock sequences have no finite expected size; usually the catalogue is "
                         "incomplete for part of the window (start later or raise mag_min)")
        share = float(np.mean(th[0] * self.background_rate[t] / lam)) if lam.size else float("nan")
        self.checks["background_share"] = share
        if share < BG_SHARE_MIN:
            notes.append(f"only {100 * share:.1f}% of the targets are background (expected count): "
                         "nearly everything is read as triggered; usually incompleteness, or a window "
                         "dominated by one sequence")
        n_outer = sum(1 for h in self.history if h[0] == "outer")
        if s.converge and self.converged and n_outer == s.max_outer:
            notes.append(f"converged only at the last allowed outer iteration ({s.max_outer}): "
                         "rerun with a larger max_outer to confirm")
        self.checks["warnings"] = notes
        # usable = what the website shows: passed the checks and p, q > 1
        self.checks["usable"] = not failed and bool(th[4] > 1 and th[6] > 1)
        if notes:
            warnings.warn("fit is valid but questionable: " + "; ".join(notes))
        if failed:
            warnings.warn("fit failed its checks: " + "; ".join(failed))
            if s.converge:
                self.converged = False

    def _change(self, metric, ref):
        """(change, new reference) for a convergence test; ref None = first evaluation.
        dphi: max |phi - ref| (ref = earlier phi array); loglik: relative change of
        -log L at the current params and background (ref = earlier -log L)."""
        if metric == "dphi":
            ch = np.inf if ref is None else float(np.max(np.abs(self.background_prob - ref)))
            return ch, self.background_prob.copy()
        f = self._func15(np.sqrt(self._theta()), self.xint0)[0]
        return (np.inf if ref is None else abs(f - ref) / abs(f)), f

    def fit(self):
        """Fixed schedule (converge=False, the original) or: outer iterations until the
        outer metric changes by < outer_tol (max max_outer), each running background/
        probability passes until the inner metric changes by < inner_tol (max max_inner)
        and then the MLE. Always starts over from the initial parameters and phi = 1."""
        s = self.settings
        self._reset()
        if s.form == "K" and not self._theta()[1] > 0:
            raise ValueError("K form: the initial parameters need p > 1 and q > 1 "
                             "(K = A (p-1)(q-1) must be positive)")
        s.use_threads()
        self.compute_bandwidths()
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
            self.neg_loglik = self._func15(np.sqrt(self._theta()), self.xint0)[0]
        if s.hessian_errors and s.optimise:
            self.hessian_errors = self.standard_errors()
        self.check_fit()
        if s.bootstrap and s.optimise:
            self.log(f" parametric bootstrap, {s.bootstrap} replicates")
            self.bootstrap = bootstrap.run(self.result(), s.bootstrap, s.bootstrap_seed, self.log)
        return self.result()

    def result(self):
        """Snapshot of the current state. Needs the bandwidths (fit or compute_bandwidths)."""
        if self.bandwidth is None:
            raise ValueError("nothing to report yet: call fit() first")
        return DeclusteringResult(
            self.catalog, self.region, self.settings, self.ind, self.params,
            self.background_prob.copy(), self.background_rate.copy(), self.bandwidth.copy(),
            self.neg_loglik, list(self.history), self.converged, self.n_likelihood_calls,
            self.ndiv_error, self.projection, self.grid, self.config, self.checks,
            self._theta()[1] if self.settings.form == "K" else None, self.hessian_errors,
            self.bootstrap)
