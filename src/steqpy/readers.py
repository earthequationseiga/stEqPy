"""Original-format input files: the catalogue and the Fortran control file (jap.in)."""
import shlex
from pathlib import Path

import numpy as np


class ListReader:
    """Mimics Fortran list-directed `read(*,*)`: each read starts on a new record, pulls
    values across as many records as needed, and drops the rest of the last one."""

    def __init__(self, text):
        self.lines = text.splitlines()
        self.pos = 0

    def read(self, k):
        vals = []
        while len(vals) < k:
            line = self.lines[self.pos].replace(",", " ")
            self.pos += 1
            vals.extend(shlex.split(line, posix=True))
        return vals[:k]


JITTER_COLS = ("lon", "lat", "mag", "t")       # column order of the rows read_catalog loads


def decimals(v):
    """Decimals of v as Python prints it (shortest repr, trailing zeros gone): 129.7 -> 1,
    129.75 -> 2, 129.0 -> 0. Like R's ETAS::catalog, so a value printed as 129.70 counts 1."""
    s = repr(float(v)).lower()
    mant, _, exp = s.partition("e")
    frac = mant.partition(".")[2].rstrip("0")
    return max(len(frac) - (int(exp) if exp else 0), 0)


def jitter_rows(rows, jitter, seed, where="catalogue"):
    """Add U(-h, h) noise to the lon, lat, mag, t columns, then re-sort by time. Used to break
    ties from rounded values (e.g. magnitudes in 0.1 steps, locations to 0.01 deg): rounding
    puts events at identical times or places, which an ETAS fit reads as extreme clustering
    (on one test catalogue, 29% of events shared a location and the spatial scale D collapsed to 0).
    h = jitter[name]: a number (same half-width for every value; 0 = untouched), or
    "precision" for lon, lat, mag: half the last decimal of each value (decimals()), as R's
    ETAS::catalog does by default for lon/lat. Caveat of that rule: a value that happens to end
    in 0 counts as coarser (129.0 in a 0.01-deg catalogue gets +-0.5 deg).
    Noise is drawn for all four columns whatever the widths, so changing one width leaves the
    other columns' noise unchanged for the same seed."""
    h = np.zeros(rows.shape)
    for c, k in enumerate(JITTER_COLS):
        v = jitter.get(k, 0.0)
        if v == "precision" and k != "t":
            h[:, c] = [0.5 * 10.0 ** -decimals(x) for x in rows[:, c]]
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and v >= 0:
            h[:, c] = v
        else:
            raise ValueError(f"jitter_{k} must be a number >= 0"
                             + ("" if k == "t" else ' or "precision"') + f", got {v!r}")
    bad = np.flatnonzero(np.diff(rows[:, 3]) < 0)
    if bad.size:                                # same check the reader makes, before re-sorting
        raise ValueError(f"{where}: Reverse data: {rows[bad[0], 3]} {rows[bad[0] + 1, 3]}")
    rng = np.random.default_rng(seed)
    rows = rows + rng.uniform(-1.0, 1.0, rows.shape) * h
    return rows[np.argsort(rows[:, 3], kind="stable")]


def read_catalog(cat_path, xmg0, zmin, tz, jitter=None, seed=0):
    """Original catalogue format: header line, then id lon lat mag t ... (t in days, sorted).
    Same filters/stop rule as the Fortran: keep mag > xmg0 (minus a float32 epsilon) and
    zmin < t <= zmin + tz. Returns raw lon, lat, mag, t arrays.
    jitter: optional {lon, lat, mag, t: half-width} (see jitter_rows), applied to every row
    before the cuts, so events can move across the magnitude and time limits."""
    rows = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
    return select_rows(rows, xmg0, zmin, tz, jitter, seed, str(cat_path))


def select_rows(rows, xmg0, zmin, tz, jitter=None, seed=0, where="catalogue"):
    """read_catalog's jitter and cuts on rows already loaded: (n, 4) lon, lat, mag, t, sorted
    by t (e.g. EventCatalog.rows())."""
    if jitter and any(jitter.values()):
        rows = jitter_rows(rows, jitter, seed, where)
    mag_cut = xmg0 - float(np.float32(0.000001))
    keep = []
    last_t = None
    for lon, lat, mag, t in rows:
        if t > tz + zmin:
            break
        if last_t is not None and t < last_t:
            raise ValueError(f"Reverse data: {last_t} {t}")
        if mag > mag_cut and t > zmin:
            keep.append((lon, lat, mag, t))
            last_t = t
        # (Fortran compares against the last *kept* event too: rejected rows are overwritten)
    if not keep:
        raise ValueError(f"{where}: no events with mag >= {xmg0} in ({zmin}, {zmin + tz}]")
    return tuple(np.array(c) for c in zip(*keep))


def read_control(infile):
    """Parse the control file (+ catalogue) like readata, without projecting: raw lon/lat
    for the events and the (open) polygon, absolute times."""
    infile = Path(infile)
    r = ListReader(infile.read_text())
    hypodata = r.read(1)[0]
    r.read(1)                                   # data format (unused)
    tz = float(r.read(1)[0])                    # length of the time interval
    r.read(1)                                   # number of events (recounted below)
    zmin = float(r.read(1)[0])                  # start time
    xmg0, tstart = map(float, r.read(2))        # magnitude threshold, target start
    npoly = int(r.read(1)[0])
    poly = np.array([list(map(float, r.read(2))) for _ in range(npoly)])
    mx, my, rtx1, rtx2, rty1, rty2 = r.read(6)
    grid = (int(mx), int(my), float(rtx1), float(rtx2), float(rty1), float(rty2))
    b = np.array(list(map(float, r.read(8))))   # (mu, A, c, alfa, p, d, q, gamma)
    npp, bwm = r.read(2)
    ioptimise = int(r.read(1)[0])
    catalog = infile.parent / hypodata
    lon, lat, mag, t = read_catalog(catalog, xmg0, zmin, tz)
    return dict(catalog=str(catalog), lon=lon, lat=lat, mag=mag, t=t,
                poly_lon=poly[:, 0], poly_lat=poly[:, 1], tz=tz, zmin=zmin, xmg0=xmg0,
                tstart=tstart, grid=grid, b=b, npp=int(npp), bwm=float(bwm),
                ioptimise=ioptimise)
