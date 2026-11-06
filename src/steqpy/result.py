"""What a fit produces: a snapshot of the estimates, plus rate maps and output files."""
import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from . import config
from . import kernels as K
from .catalog import Catalog, Region
from .params import ETASParams, Settings


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

    # -- declustering outputs ------------------------------------------------------------
    @cached_property
    def final_background_prob(self):
        """phi_i at the final parameters. background_prob (probs.dat) is from before the last
        MLE step, as in the original; the outputs below use the final parameters, as the
        original's pmatr.dat does, so that phi_i + sum_j rho_ij = 1."""
        c = self.catalog
        out = np.empty(len(c))
        self.settings.use_threads()
        self.settings.kernel("probcalc")(np.sqrt(self.params.to_array()), self.background_rate,
                                         c.x, c.y, c.t, c.m, out)
        return out

    def triggering_probs(self, i):
        """rho_ij, j < i: the probability that earlier event j triggered event i, in an array
        of length i. Computed on demand, so all pairs need not be held in memory."""
        c = self.catalog
        if not 0 <= i < len(c):
            raise IndexError(f"event {i} out of range for {len(c)} events")
        rates = np.empty(i)
        bk = K.trigger_terms(i, np.sqrt(self.params.to_array()), self.background_rate,
                             c.x, c.y, c.t, c.m, rates)
        return rates / (bk + rates.sum())

    def sample_background(self, seed=None):
        """One random declustered catalogue: a boolean mask over all events, True = background
        (Algorithm 2 of Zhuang et al. 2002). Includes non-targets: combine with
        `ind == 1` to keep only the target events."""
        phi = self.final_background_prob
        return np.random.default_rng(seed).random(len(phi)) < phi

    def sample_parents(self, seed=None):
        """One random family tree: each event's parent index, -1 = background (Algorithm 3 of
        Zhuang et al. 2004). The same seed gives the same tree at any thread count."""
        c = self.catalog
        u = np.random.default_rng(seed).random(len(c))
        self.settings.use_threads()
        return self.settings.kernel("sample_parents")(
            np.sqrt(self.params.to_array()), self.background_rate, c.x, c.y, c.t, c.m, u)

    # -- rate maps and output files ------------------------------------------------------
    def rates(self, gx, gy):
        """(len(gx)*len(gy), 3): background rate, unweighted kernel density, total
        intensity at t_end, x-major. gx, gy in model coordinates."""
        c, r = self.catalog, self.region
        self.settings.use_threads()
        return self.settings.kernel("outrates_grid")(
            np.sqrt(self.params.to_array()), c.x, c.y, c.t, c.m, self.background_prob,
            self.bandwidth, r.t_end, r.t_start, np.asarray(gx, float), np.asarray(gy, float))

    def rates_lonlat(self, lon, lat):
        """rates() at arbitrary lon/lat points (1-D arrays of equal length)."""
        c, r = self.catalog, self.region
        px, py = self.projection(lon, lat)
        self.settings.use_threads()
        return self.settings.kernel("outrates_points")(
            np.sqrt(self.params.to_array()), c.x, c.y, c.t, c.m, self.background_prob,
            self.bandwidth, r.t_end, r.t_start, px, py)

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
