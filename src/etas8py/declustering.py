"""Object-oriented API for stochastic declustering (Zhuang, Ogata & Vere-Jones 2002):
space-time ETAS model with a variable-bandwidth kernel background.

    from etas8py import StochasticDeclustering
    sd = StochasticDeclustering.from_input_file("data/jma/jap.in", threads=8)
    sd.fit()
    sd.params                 # ETASParams (MLE)
    sd.background_prob        # P(event i is a background event), per event
    sd.rates(gx, gy)          # background / kernel density / total intensity on a grid

Pieces are separate so each can be swapped or used alone: Catalog (events), Region (study
polygon + target window), ETASParams (the 8 model parameters), Settings (algorithm
hyperparameters), StochasticDeclustering (the estimation steps). The numerics live in
`kernels` (Numba); with default Settings and fast_xint=False, threads=0 this reproduces the
original Fortran bit for bit (validated in tests/test_declustering.py against run.Model).
"""
from dataclasses import astuple, dataclass

import numpy as np

from . import kernels as K
from .davidn import davidn

# Original schedule: outer iteration k = 1..11 does (12 - k) background/probability passes,
# then one maximum-likelihood fit. 66 passes in total; there is no convergence test.
FULL_SCHEDULE = tuple(12 - k for k in range(1, 12))


@dataclass
class Catalog:
    """Time-sorted events in model coordinates. x = lon * cos(lat of region centroid) and
    y = lat (flat local projection, degrees); t = days since the catalog start;
    m = magnitude minus the cutoff magnitude (so m >= 0)."""
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
    """Study region. polygon_x/_y: vertices in model coordinates, closed (last == first).
    Events inside the polygon with t_start <= t <= t_end are the *targets* whose likelihood
    is fitted; earlier/outside events still trigger (they enter the intensity)."""
    polygon_x: np.ndarray
    polygon_y: np.ndarray
    t_start: float
    t_end: float

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
    mu: background scale; A, alpha: productivity; c, p: Omori-Utsu time decay;
    D, q, gamma: spatial spread (magnitude-scaled) and its decay."""
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

    def __post_init__(self):
        if any(v <= 0 for v in astuple(self)):
            raise ValueError("all ETAS parameters must be positive")


@dataclass(frozen=True)
class Settings:
    """Algorithm hyperparameters. Defaults = the original code / JMA example."""
    npp: int = 5                    # background bandwidth = distance to npp-th nearest event
    bwm: float = 0.05               # ...floored at bwm (degrees)
    schedule: tuple = FULL_SCHEDULE  # bkgd/prob passes per outer iteration (each + 1 MLE)
    optimise: bool = True           # False: keep the initial ETAS parameters fixed
    ndiv: int = 1000                # pieces per polygon edge in the spatial integrals
    tol: float = 1e-6               # optimiser stopping tolerance (gradient/step/rel. f)
    max_sweeps: int = 10            # optimiser: max sweeps of 8 quasi-Newton steps
    fast_xint: bool = True          # fused polygon integrals (same result to ~1e-12)
    threads: int = 0                # 0 = serial loops; >0 = Numba threads (bit-identical)

    def __post_init__(self):
        if self.npp < 1 or self.bwm <= 0 or self.ndiv < 1 or self.tol <= 0:
            raise ValueError("need npp >= 1, bwm > 0, ndiv >= 1, tol > 0")
        if self.max_sweeps < 1 or not all(int(k) >= 0 for k in self.schedule):
            raise ValueError("need max_sweeps >= 1 and non-negative schedule entries")
        if self.threads < 0:
            raise ValueError("threads must be >= 0")


class StochasticDeclustering:
    """Alternates (a) background estimation given the background probabilities and
    (b) ETAS maximum likelihood given the background, then (c) probabilities given both.
    State after fit(): params, background_prob, background_rate, bandwidth."""

    def __init__(self, catalog, region, init_params, settings=Settings(), log=None):
        self.catalog = catalog
        self.region = region
        self.settings = settings
        self.params = init_params
        self.log = log or (lambda msg: None)
        self.ind = region.target_flags(catalog)
        n = len(catalog)
        self.background_prob = np.ones(n)       # zprob: start as "everything is background"
        self.background_rate = np.zeros(n)      # zbkgd: u(x_i, y_i) per unit time
        self.bandwidth = None                   # zbandw
        self.xint0 = 0.0                        # integral of the background over the region
        self.n_likelihood_calls = 0
        if settings.threads > 0:
            import numba
            numba.set_num_threads(settings.threads)
            numba.set_parallel_chunksize(16)   # balances xlamb's O(i) per-event cost

    @classmethod
    def from_input_file(cls, path, log=None, **overrides):
        """Build from an original-format control file (e.g. jap.in). npp, bwm and optimise
        come from the file; any Settings field can be overridden by keyword. The file's
        rate grid is kept as self.grid."""
        from .run import readata
        d = readata(path)
        file_settings = dict(npp=d["npp"], bwm=d["bwm"], optimise=d["ioptimise"] == 1)
        settings = Settings(**{**file_settings, **overrides})
        cat = Catalog(d["xx"], d["yy"], d["zz"], d["zmg"])
        reg = Region(d["tx"], d["ty"], d["tstart"], d["tz"])
        sd = cls(cat, reg, ETASParams.from_array(d["b"]), settings, log)
        sd.grid = d["grid"]
        return sd

    # -- the estimation steps (each usable on its own) ----------------------------------
    def _k(self, name):
        return getattr(K, name + "_par") if self.settings.threads > 0 else getattr(K, name)

    def compute_bandwidths(self):
        c, s = self.catalog, self.settings
        self.bandwidth = K.bandwcalc(c.x, c.y, s.npp, s.bwm)

    def update_background(self):
        """u(x_i, y_i) = sum_j phi_j g_{h_j}(r_ij) / T and its integral over the region."""
        c, r = self.catalog, self.region
        self.xint0 = self._k("bkgdcalc")(c.x, c.y, self.background_prob, self.bandwidth,
                                         r.polygon_x, r.polygon_y, r.npoly, r.t_end,
                                         r.t_start, self.background_rate, self.settings.ndiv)

    def update_probabilities(self):
        """phi_i = mu u_i / lambda_i."""
        c = self.catalog
        self._k("probcalc")(np.sqrt(self.params.to_array()), self.background_rate, c.x, c.y,
                            c.t, c.m, self.background_prob)

    def neg_log_likelihood(self, sqrt_params):
        """(-log L, gradient w.r.t. sqrt(params)) with the background held fixed."""
        c, r, s = self.catalog, self.region, self.settings
        grad = np.empty(8)
        f = self._k("func15")(np.asarray(sqrt_params, dtype=np.float64), self.ind,
                              self.background_rate, self.xint0, c.x, c.y, c.t, c.m,
                              r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start, grad,
                              s.fast_xint, s.ndiv)
        self.n_likelihood_calls += 1
        return f, list(grad)

    def maximise(self):
        """Quasi-Newton (DFP) MLE on sqrt(params), which keeps every parameter positive."""
        x = list(np.sqrt(self.params.to_array()))
        davidn(x, lambda b, log: self.neg_log_likelihood(b), self.log,
               self.settings.tol, self.settings.max_sweeps)
        self.params = ETASParams.from_array(np.array(x) ** 2)

    def fit(self):
        self.compute_bandwidths()
        for k, reps in enumerate(self.settings.schedule, 1):
            self.log(f" Iteration {k}")
            for _ in range(reps):
                self.update_background()
                self.update_probabilities()
            if self.settings.optimise:
                self.maximise()
                self.log(f" mle = {self.params}")
        return self

    # -- outputs ------------------------------------------------------------------------
    def rates(self, gx, gy):
        """(len(gx)*len(gy), 3): background rate, unweighted kernel density, total
        intensity at t_end, x-major. gx, gy in model coordinates."""
        c, r = self.catalog, self.region
        return self._k("outrates_grid")(np.sqrt(self.params.to_array()), c.x, c.y, c.t, c.m,
                                        self.background_prob, self.bandwidth, r.t_end,
                                        r.t_start, np.asarray(gx, float),
                                        np.asarray(gy, float))


__all__ = ["Catalog", "Region", "ETASParams", "Settings", "StochasticDeclustering",
           "FULL_SCHEDULE"]
