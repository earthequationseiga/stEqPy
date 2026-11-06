"""Numbers and figure data for docs/algorithm_explainer.tex, from the default flat JMA run."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from steqpy import kernels as K
from steqpy.declustering import StochasticDeclustering as S

sd = S.from_config(ROOT / "configs/jma.toml")
c, r = sd.catalog, sd.region
P = np.loadtxt(ROOT / "results/cfg_default_flat/probs.dat")
phi, h, u = P[:, 1], P[:, 2], P[:, 3]
tg = sd.ind == 1      # ind is 1 / 0 / -1: only 1 is a target
th = {k: float(v) for k, v in zip("mu A c alpha p D q gamma".split(), [
    0.5506075916, 0.1657633799, 0.02961548263, 1.657926661, 1.153401021,
    0.001833837161, 1.950547676, 1.067036079])}
print("N", len(phi), "targets", tg.sum(), "t range", c.t.min(), c.t.max(), "maxM", c.m.max() + 4.5)
print("sum phi all", phi.sum(), "targets", phi[tg].sum(), "frac", phi[tg].mean())
print("targets phi>.9", (phi[tg] > .9).sum(), "<.1", (phi[tg] < .1).sum(), "between",
      ((phi[tg] >= .1) & (phi[tg] <= .9)).sum())
print("hist10", np.histogram(phi[tg], bins=np.linspace(0, 1, 11))[0].tolist())
print("h pct", np.percentile(h, [1, 50, 99]) * 111.195, "at floor", (h <= .05).sum())
xint0 = K.bkgdcalc(c.x, c.y, phi, h, r.polygon_x, r.polygon_y, r.npoly, r.t_end, r.t_start,
                   np.empty(len(phi)))
print("xint0", xint0, "mu*xint0", th["mu"] * xint0, "(MLE identity: approx sum phi targets)")


def pair(i, j):
    """Contribution of earlier event j to the intensity at event i."""
    dt = c.t[i] - c.t[j]
    s = th["D"] * np.exp(th["gamma"] * c.m[j])
    r2 = (c.x[i] - c.x[j]) ** 2 + (c.y[i] - c.y[j]) ** 2
    return (th["A"] * np.exp(th["alpha"] * c.m[j])
            * (th["p"] - 1) / th["c"] * (1 + dt / th["c"]) ** -th["p"]
            * (th["q"] - 1) / (np.pi * s) * (1 + r2 / s) ** -th["q"]), dt, np.sqrt(r2) * 111.195


def show(i):
    print(f"event {i + 1} M{c.m[i] + 4.5:.1f} t={c.t[i]:.2f} phi={phi[i]:.4f} u={u[i]:.5f} "
          f"mu*u={th['mu'] * u[i]:.5f} lambda(mu u/phi)={th['mu'] * u[i] / phi[i]:.5f}")
    k = np.array([pair(i, j)[0] for j in range(i)])
    lam = th["mu"] * u[i] + k.sum()
    print(f"  lambda recomputed {lam:.5f} phi recomputed {th['mu'] * u[i] / lam:.4f}")
    for j in np.argsort(-k)[:3]:
        v, dt, d = pair(i, j)
        print(f"  parent {j + 1} M{c.m[j] + 4.5:.1f} dt={dt:.3f} d d={d:.1f} km kappa={v:.5f} "
              f"rho={v / lam:.4f}")


big = np.where(tg & (c.m >= 2.5))[0]
print("big target events (M>=7):", [(int(b) + 1, round(float(c.m[b] + 4.5), 1), round(float(c.t[b]))) for b in big][:12])
for b in big[:1]:
    aft = np.where(tg & (c.t > c.t[b]) & (c.t < c.t[b] + 5))[0]
    for a in aft[[0, len(aft) // 2]]:
        show(a)
show(np.where(tg & (phi > .95))[0][200])
mid = np.where(tg & (phi > .4) & (phi < .6))[0]
show(mid[len(mid) // 2])

# Cumulative counts: raw targets vs expected background (phi-weighted), thinned for plotting.
t = c.t[tg]
cum, cumb = np.arange(1, tg.sum() + 1), np.cumsum(phi[tg])
idx = np.unique(np.r_[np.linspace(0, len(t) - 1, 300).astype(int)])
np.savetxt(ROOT / "docs/explainer_cum.dat", np.c_[1926 + t[idx] / 365.25, cum[idx], cumb[idx]],
           header="year all background", comments="", fmt="%.3f %d %.1f")
for dm in (0, 1, 2, 3, c.m.max()):
    print(f"M{dm + 4.5:.1f}: children {th['A'] * np.exp(th['alpha'] * dm):.2f} "
          f"scale km {np.sqrt(th['D'] * np.exp(th['gamma'] * dm)) * 111.195:.1f}")
