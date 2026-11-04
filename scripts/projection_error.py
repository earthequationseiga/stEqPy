"""Distance error of the model projections on a real catalogue, against great-circle
distance: every event pair closer than 1 deg (the scales the kernels see), and the
npp-th-neighbour bandwidths. Usage: python -I scripts/projection_error.py [data/jma/jap.in]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from etas8py import geo  # noqa: E402
from etas8py.run import read_control  # noqa: E402

NPP = 5


def pair_stats(lon, lat, x, y, gc_max=1.0, block=200):
    """Ratios projected/great-circle for pairs with gc < gc_max, and per-event NPP-th
    neighbour distance in both metrics."""
    n = len(lon)
    ratios, nn_p, nn_g = [], np.empty(n), np.empty(n)
    for s in range(0, n, block):
        i = slice(s, min(s + block, n))
        g = geo.great_circle_deg(lon[i, None], lat[i, None], lon[None], lat[None])
        p = np.hypot(x[i, None] - x[None], y[i, None] - y[None])
        nn_g[i] = np.partition(g, NPP, axis=1)[:, NPP]     # column 0 is the event itself
        nn_p[i] = np.partition(p, NPP, axis=1)[:, NPP]
        rows = np.arange(i.start, i.stop)[:, None]
        m = (g < gc_max) & (g > 0) & (np.arange(n)[None] > rows)
        ratios.append(p[m] / g[m])
    return np.concatenate(ratios), nn_p / nn_g


if __name__ == "__main__":
    c = read_control(sys.argv[1] if len(sys.argv) > 1 else "data/jma/jap.in")
    lon, lat = c["lon"], c["lat"]
    q = [0, 1, 50, 99, 100]
    for kind in ("flat", "laea"):
        pr = geo.Projection(kind, c["poly_lon"], c["poly_lat"])
        x, y = pr(lon, lat)
        r, bw = pair_stats(lon, lat, x, y)
        print(f"{kind}: centre lon {pr.lon0:.2f} lat {pr.lat0:.2f}; {len(r)} pairs < 1 deg")
        print("  pair distance ratio  min/p1/median/p99/max "
              + " ".join(f"{v:.4f}" for v in np.percentile(r, q)))
        print(f"  NPP={NPP} bandwidth ratio  min/p1/median/p99/max "
              + " ".join(f"{v:.4f}" for v in np.percentile(bw, q)))
