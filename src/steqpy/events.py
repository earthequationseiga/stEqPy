"""An earthquake catalogue as read, in natural units: the starting point of a declustering run.

    from steqpy import EventCatalog, StochasticDeclustering
    events = EventCatalog.from_csv("my_catalogue.csv")     # or .from_arrays(...), .from_etas(...)
    events.completeness("maxc")                            # Mc, to choose mag_min
    sd = StochasticDeclustering.from_catalog(events, polygon, "2000-01-01", "2024-12-31", 5.2)
    result = sd.fit()

Catalog (catalog.py) is the fitting form of the same events: projected, cut at mag_min, with
magnitudes relative to it. EventCatalog keeps longitude/latitude, the magnitudes as given,
depth (not used by the model), and real times when they are known.
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import completeness

DAY = np.timedelta64(86_400_000, "ms")
# CSV header names recognised, compared lower-cased with everything but letters and digits
# removed ("Depth (km)" -> "depthkm"); earlier names win. Same lists as the website.
ALIASES = {
    "time": ("time", "origintime", "origintimeutc", "timeutc", "datetime", "eventtime", "timestamp", "date"),
    "lon": ("longitude", "lon", "long", "lng"),
    "lat": ("latitude", "lat"),
    "mag": ("magnitude", "mag", "mw", "ml", "mb", "ms", "m"),
    "depth": ("depth", "depthkm", "dep"),
}


def _parse_time(s):
    """ISO-like date-time text as UTC milliseconds ("2016-09-12 11:32:54", "...T...Z",
    a bare date). Times without a zone are UTC; numeric offsets are not supported."""
    s = s.strip().replace(" ", "T")
    if s.endswith(("Z", "z")):
        s = s[:-1]
    if re.search(r"[+-]\d{2}:?\d{2}$", s[10:]):
        raise ValueError(f"time zone offsets are not supported: {s!r} (give UTC)")
    return np.datetime64(s, "ms")


@dataclass(eq=False)
class EventCatalog:
    """Events sorted by time: lon, lat (degrees), t (days since t0), mag, depth (km, NaN =
    unknown). t0: numpy datetime64 of t = 0, or None when the times were given as plain days."""
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    mag: np.ndarray
    depth: np.ndarray = None
    t0: np.datetime64 = None

    def __post_init__(self):
        cols = [np.asarray(v, dtype=np.float64) for v in (self.lon, self.lat, self.t, self.mag)]
        n = len(cols[0])
        depth = np.full(n, np.nan) if self.depth is None else np.asarray(self.depth, dtype=np.float64)
        if not all(len(c) == n for c in (*cols, depth)):
            raise ValueError("lon, lat, t, mag (and depth) must have the same length")
        if n == 0:
            raise ValueError("no events")
        for name, c, lo, hi in (("lon", cols[0], -180, 180), ("lat", cols[1], -90, 90),
                                ("t", cols[2], -np.inf, np.inf), ("mag", cols[3], -5, 12)):
            bad = ~(np.isfinite(c) & (c >= lo) & (c <= hi))
            if bad.any():
                raise ValueError(f"{name}: event {int(np.argmax(bad))} is {c[bad][0]!r}, "
                                 f"need a finite number in [{lo}, {hi}]")
        o = np.argsort(cols[2], kind="stable")
        self.lon, self.lat, self.t, self.mag = (c[o] for c in cols)
        self.depth = depth[o]

    def __len__(self):
        return len(self.t)

    # -- building -------------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, lon, lat, time, mag, depth=None, t0=None):
        """time: datetime64 values or ISO strings (then t0 defaults to the first event's
        time), or plain numbers taken as days (t0 stays None unless given)."""
        time = np.asarray(time)
        if time.dtype.kind in "MUSO":
            tt = time.astype("datetime64[ms]") if time.dtype.kind == "M" else \
                np.array([_parse_time(str(s)) for s in time])
            t0 = tt.min() if t0 is None else np.datetime64(t0, "ms")
            return cls(lon, lat, (tt - t0) / DAY, mag, depth, t0)
        return cls(lon, lat, time.astype(np.float64), mag, depth,
                   None if t0 is None else np.datetime64(t0, "ms"))

    @classmethod
    def from_csv(cls, path, delimiter=None):
        """CSV with a header row naming the time, longitude, latitude and magnitude columns
        (any order; common names recognised, see ALIASES) and optionally depth; other columns
        are ignored. delimiter: default the one of , tab ; that splits the header most."""
        text = Path(path).read_text(encoding="utf-8-sig")
        head = text.split("\n", 1)[0]
        delimiter = delimiter or max((",", "\t", ";"), key=lambda s: len(head.split(s)))
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        names = [re.sub(r"[^a-z0-9]", "", h.lower()) for h in rows[0]]
        col = {}
        for field, aliases in ALIASES.items():
            for a in aliases:
                if a in names:
                    col[field] = names.index(a)
                    break
        missing = [f for f in ("time", "lon", "lat", "mag") if f not in col]
        if missing:
            raise ValueError(f"{path}: no column for {', '.join(missing)} in header {rows[0]}")
        vals = {f: [] for f in ("time", "lon", "lat", "mag", "depth")}
        for line, r in enumerate(rows[1:], start=2):
            if not any(v.strip() for v in r):
                continue
            try:
                vals["time"].append(_parse_time(r[col["time"]]))
                for f in ("lon", "lat", "mag"):
                    vals[f].append(float(r[col[f]]))
                d = r[col["depth"]].strip() if "depth" in col else ""
                vals["depth"].append(float(d) if d else np.nan)
            except (ValueError, IndexError) as e:
                raise ValueError(f"{path}, line {line}: {e}") from None
        return cls.from_arrays(vals["lon"], vals["lat"], np.array(vals["time"]), vals["mag"], vals["depth"])

    @classmethod
    def from_etas(cls, path, t0=None):
        """Zhuang's catalogue format (header line, then id lon lat mag t ..., t in days)."""
        d = np.loadtxt(path, skiprows=1, usecols=(1, 2, 3, 4), ndmin=2)
        return cls(d[:, 0], d[:, 1], d[:, 3], d[:, 2], None,
                   None if t0 is None else np.datetime64(t0, "ms"))

    def to_etas(self, path):
        """Write in Zhuang's format, values exact (re-reading gives the same floats)."""
        with open(path, "w") as fh:
            fh.write(f"'id lon lat mag t(days since {self.t0 if self.t0 is not None else 't0'})'\n")
            for i in range(len(self)):
                fh.write(f"{i + 1} {float(self.lon[i])!r} {float(self.lat[i])!r} "
                         f"{float(self.mag[i])!r} {float(self.t[i])!r}\n")

    # -- using ----------------------------------------------------------------------------
    def days(self, when):
        """A time as days since t0: numbers pass through; dates (text or datetime64) need t0."""
        if isinstance(when, (int, float, np.integer, np.floating)):
            return float(when)
        if self.t0 is None:
            raise ValueError("this catalogue's times are plain days (no t0): give times as days")
        w = _parse_time(when) if isinstance(when, str) else np.datetime64(when, "ms")
        return float((w - self.t0) / DAY)

    def rows(self):
        """(n, 4) array lon, lat, mag, t: the column order the readers work on."""
        return np.column_stack([self.lon, self.lat, self.mag, self.t])

    def select(self, mask):
        """The events where mask is true (a boolean array), as a new EventCatalog."""
        mask = np.asarray(mask, dtype=bool)
        return EventCatalog(self.lon[mask], self.lat[mask], self.t[mask], self.mag[mask],
                            self.depth[mask], self.t0)

    def between(self, start=None, end=None):
        """Events with start <= t <= end (days or dates; None = open)."""
        lo = -np.inf if start is None else self.days(start)
        hi = np.inf if end is None else self.days(end)
        return self.select((self.t >= lo) & (self.t <= hi))

    def completeness(self, method="maxc", dm=None, start=None, end=None):
        """Magnitude of completeness of the events in [start, end], by "maxc" (MAXC + 0.2),
        "gft", "mbs" or "cv" (see completeness). dm: magnitude bin (default: from the data)."""
        return completeness.mc(self.between(start, end).mag, method, dm)

    def mc_over_time(self, method="cv", step=365.25 / 4, window=365.25 / 2, dm=None):
        """Mc in windows of `window` days ending every `step` days (default: every 3 months
        from the 6 months before, as the website), plus event counts per step. Returns a dict:
        bin_start, counts (events in [bin_start, bin_start + step)), t (window ends), mc (NaN:
        too few events to judge), n (events per window). Times in days since t0."""
        m = np.round(self.mag, 6)
        dm = completeness.bin_width(m) if dm is None else dm
        t0, t1 = self.t.min(), self.t.max()
        nb = max(1, int(np.ceil((t1 - t0) / step)))
        starts = t0 + step * np.arange(nb)
        counts = np.histogram(self.t, np.append(starts, max(t1, starts[-1]) + step))[0]
        ends = starts[1:] + step                    # first window ends after two steps
        mcs, ns = [], []
        for e in ends:
            sel = (self.t > e - window) & (self.t <= e)
            ns.append(int(sel.sum()))
            mcs.append(completeness.mc(m[sel], method, dm) if sel.any() else np.nan)
        return {"bin_start": starts, "counts": counts, "t": ends, "mc": np.array(mcs, float),
                "n": np.array(ns)}

    def plot_completeness(self, method="cv", step=365.25 / 4, window=365.25 / 2, ax=None):
        """The website's completeness figure: events per step (bars, log scale) and Mc over
        time (dots, right axis), see mc_over_time. Needs matplotlib. Returns the Axes."""
        from . import plotting
        return plotting.completeness(self, method, step, window, ax)

    def b_value(self, mc, dm=None):
        """Aki-Utsu b of the events with mag >= mc, its standard error, and their count."""
        m = np.round(self.mag, 6)
        return completeness.b_value(m, mc, completeness.bin_width(m) if dm is None else dm)
