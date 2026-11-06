"""The numbers: model parameters (ETASParams) and algorithm hyperparameters (Settings)."""
import warnings
from dataclasses import astuple, dataclass

import numpy as np

from . import geo
from . import kernels as K

# Original schedule: outer iteration k = 1..11 does (12 - k) background/probability passes,
# then one maximum-likelihood fit. 66 passes in total; there is no convergence test.
FULL_SCHEDULE = tuple(12 - k for k in range(1, 12))
METRICS = ("loglik", "dphi")


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
    "default"; preset "original" = converge=False, quadrature="ndiv", fast_xint=False,
    ndiv_check=False."""
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
    quadrature: str = "sgl"         # polygon integrals: "sgl" (see kernels) or "ndiv" (original)
    sgl_nodes: int = 48             # sgl: nodes per polygon edge, ETAS kernel and derivatives
    sgl_nodes_background: int = 64  # sgl: nodes per polygon edge, Gaussian background kernel
    ndiv: int = 1000                # quadrature "ndiv": pieces per polygon edge
    ndiv_check: bool = True         # one-off: -log L change when nodes or ndiv are doubled
    ndiv_check_tol: float = 1e-6    # ...warn above this relative change
    fast_xint: bool = True          # quadrature "ndiv": fused integrals (same to ~1e-12)
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
        if self.quadrature not in ("sgl", "ndiv"):
            raise ValueError("quadrature must be 'sgl' or 'ndiv'")
        if (self.npp < 1 or self.ndiv < 1 or self.sgl_nodes < 1 or self.sgl_nodes_background < 1
                or self.max_sweeps < 1 or self.max_outer < 1 or self.max_inner < 1
                or self.chunksize < 1):
            raise ValueError("need npp, ndiv, sgl_nodes, sgl_nodes_background, max_sweeps, "
                             "max_outer, max_inner, chunksize >= 1")
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

    def kernel(self, name):
        """The Numba routine `name`: its parallel version if threads are on."""
        return getattr(K, name + "_par") if self.threads > 0 else getattr(K, name)

    def use_threads(self):
        # process-global: the last fit()/rates() call decides for every instance
        if self.threads > 0:
            import numba
            numba.set_num_threads(self.threads)
            numba.set_parallel_chunksize(self.chunksize)
