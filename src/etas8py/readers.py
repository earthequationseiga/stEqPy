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


def read_catalog(cat_path, xmg0, zmin, tz):
    """Original catalogue format: header line, then id lon lat mag t ... (t in days, sorted).
    Same filters/stop rule as the Fortran: keep mag > xmg0 (minus a float32 epsilon) and
    zmin < t <= zmin + tz. Returns raw lon, lat, mag, t arrays."""
    rows = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
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
        raise ValueError(f"{cat_path}: no events with mag >= {xmg0} in ({zmin}, {zmin + tz}]")
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
