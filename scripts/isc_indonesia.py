"""Indonesian test catalogue: ISC Bulletin events with a reported Mw >= 4, depth <= 50 km,
1963-2016, box 18.10S-4.61N, 93.08-138.47E (the region of the Z-value study, Nat Hazards 2019,
doi:10.1007/s11069-019-03689-y; that study converted mb/MS to Mw, here only events that already
have an Mw are kept, so no conversion).

Download (one file per year, ISC web search, CATCSV, req_mag_type=MW keeps events that have an
MW of >= 4 from any agency), into data/isc_indonesia/isc_mw_YYYY.html:
    https://www.isc.ac.uk/cgi-bin/web-db-run?request=COMPREHENSIVE&out_format=CATCSV
      &searchshape=RECT&bot_lat=-18.10&top_lat=4.61&left_lon=93.08&right_lon=138.47
      &start_year=Y&start_month=1&start_day=1&start_time=00:00:00
      &end_year=Y&end_month=12&end_day=31&end_time=23:59:59
      &max_dep=50&min_mag=4&req_mag_type=MW&req_mag_agcy=Any

Location: the ISC prime hypocentre (first line of each event). Magnitude: an event can have
several Mw; take GCMT's, else NEIC's, else the first other agency's (types in MW_TYPES).
Sources (SOURCES): --isc_mw (default) as above; --isc_mixed from isc_all_YYYY.html (same
search with req_mag_type=Any), events without an Mw get one from Ms, else mb (ISC's, else
NEIC's, else another agency's) with the study's regressions; --usgs_mw / --usgs_all from USGS
ComCat CSVs (see read_usgs). Writes data/<folder>/indonesia_<source>4_<y0>-<y1>.etas (steqpy /
Zhuang catalogue format, t in days since Jan 1 of y0) and, with --plot (needs cartopy),
figures/indonesia/events_<same>.png. --compare: magnitude counts and events per year of every
source with data, figures/indonesia/compare_<y0>-<y1>.png.
    python -I scripts/isc_indonesia.py [--isc_mw|--isc_mixed|--usgs_mw|--usgs_all]
        [--years 1963 2016] [--plot] [--compare]
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "isc_indonesia"
PRIORITY = ("GCMT", "NEIC")
# moment magnitudes (NEIC's W-phase, body-, centroid-, regional-wave inversions); not Mwp
# (P-wave proxy) or MW(mb) (converted from mb)
MW_TYPES = ("MW", "MWW", "MWB", "MWC", "MWR")
BOX = (93.08, 138.47, -18.10, 4.61)
# --mixed: events without an Mw get one from Ms or mb with the study's regressions (its Fig. 2).
# Both quadratics have a minimum (mb 3.87 -> Mw 4.68, Ms 1.67 -> Mw 4.57) and rise again below
# it, so they are used only above the minimum: no converted Mw is below 4.57-4.68.
MS_TO_MW = (lambda ms: 0.09 * ms ** 2 - 0.3 * ms + 4.82, 0.3 / (2 * 0.09))
MB_TO_MW = (lambda mb: 0.3 * mb ** 2 - 2.32 * mb + 9.17, 2.32 / (2 * 0.3))
ISC_FIRST = ("ISC", "NEIC")     # agency order for mb and Ms


def first(mags, types, priority):
    """(agency, value) of the first magnitude of the given types, by agency priority."""
    found = {}
    for agency, typ, val in mags:
        if typ in types and val and agency not in found:
            found[agency] = float(val)
    if not found:
        return None
    agency = next((a for a in priority if a in found), next(iter(found)))
    return agency, found[agency]


def parse(path, mixed=False):
    """Rows (time, lon, lat, depth, mw, source) of one downloaded year; source is the Mw's
    agency, or "Ms <agency>" / "mb <agency>" for a converted one."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if not re.match(r"^\s*\d+\s*,", line):
            continue
        f = [s.strip() for s in line.split(",")]
        if len(f) < 12 or not f[7]:      # not an event line, or no depth
            continue
        # EVENTID,TYPE,AUTHOR,DATE,TIME,LAT,LON,DEPTH,DEPFIX, then (AUTHOR,TYPE,MAG) repeated
        raw = [(f[i], f[i + 1], f[i + 2]) for i in range(9, len(f) - 2, 3)]
        mags = [(a, t.upper(), v) for a, t, v in raw]
        pick = first(mags, MW_TYPES, PRIORITY)
        if pick:
            src, mw = pick
        elif mixed:
            # case-sensitive types: "mB" (broadband) and "mb1" etc. are other scales
            ms = first(raw, ("MS", "Ms"), ISC_FIRST)
            mb = first(raw, ("mb",), ISC_FIRST)
            if ms and ms[1] >= MS_TO_MW[1]:
                src, mw = "Ms " + ms[0], MS_TO_MW[0](ms[1])
            elif mb and mb[1] >= MB_TO_MW[1]:
                src, mw = "mb " + mb[0], MB_TO_MW[0](mb[1])
            else:
                continue
        else:
            continue
        t = datetime.fromisoformat(f[3] + " " + f[4])    # fraction of a second sometimes absent
        out.append((t, float(f[6]), float(f[5]), float(f[7]), round(mw, 2), src))
    return out


def read_usgs(path, mw_only):
    """Rows like parse's from a USGS ComCat CSV (preferred magnitude only; source = magType).
    Download: earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=Y-01-01
    &endtime=Y+1-01-01&minlatitude=-18.10&maxlatitude=4.61&minlongitude=93.08
    &maxlongitude=138.47&maxdepth=50&minmagnitude=4 into data/usgs_indonesia/usgs_Y.csv."""
    import csv
    out = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            typ = r["magType"].lower()
            if r["type"] != "earthquake" or (mw_only and not typ.startswith("mw")):
                continue
            t = datetime.fromisoformat(r["time"].rstrip("Z"))
            out.append((t, float(r["longitude"]), float(r["latitude"]), float(r["depth"]),
                        float(r["mag"]), "Mw" if typ.startswith("mw") else typ))
    return out


# name: (data folder, file glob, reader, label)
SOURCES = {
    "isc_mw": ("isc_indonesia", "isc_mw_*.html", lambda p: parse(p), "ISC, reported Mw"),
    "isc_mixed": ("isc_indonesia", "isc_all_*.html", lambda p: parse(p, True),
                  "ISC, Mw or Ms/mb converted to Mw"),
    "usgs_mw": ("usgs_indonesia", "usgs_*.csv", lambda p: read_usgs(p, True), "USGS, Mw types"),
    "usgs_all": ("usgs_indonesia", "usgs_*.csv", lambda p: read_usgs(p, False),
                 "USGS, preferred magnitude (mostly mb)"),
}


def load(source, y0, y1):
    """Rows of one source in years y0..y1, depth <= 50 km, magnitude >= 4, sorted by time."""
    folder, pattern, reader, _ = SOURCES[source]
    rows = []
    for p in sorted((ROOT / "data" / folder).glob(pattern)):
        if y0 <= int(re.search(r"(\d{4})", p.name).group(1)) <= y1:
            rows += reader(p)
    rows.sort(key=lambda r: r[0])
    # the ISC search passes an event if any magnitude of the type is >= 4; the chosen one may not be
    return [r for r in rows if r[3] <= 50 and r[4] >= 4.0 and y0 <= r[0].year <= y1]


def main():
    args = sys.argv[1:]
    source = next((a[2:] for a in args if a[2:] in SOURCES), "isc_mw")
    y0, y1 = (int(args[args.index("--years") + 1]), int(args[args.index("--years") + 2])) \
        if "--years" in args else (1963, 2016)
    if "--compare" in args:
        return compare(y0, y1)
    rows = load(source, y0, y1)
    t0 = datetime(y0, 1, 1)
    tag = f"{source}4_{y0}-{y1}"
    t = np.array([(r[0] - t0).total_seconds() / 86400 for r in rows])
    lon, lat, dep, mw = (np.array([r[k] for r in rows]) for k in (1, 2, 3, 4))
    ag = np.array([r[5] for r in rows])
    out = ROOT / "data" / SOURCES[source][0] / f"indonesia_{tag}.etas"
    with out.open("w") as fh:
        fh.write(f"'{SOURCES[source][3]} >= 4, depth <= 50 km, t days since {t0:%Y-%m-%d}'\n")
        for i, r in enumerate(rows):
            fh.write(f"{i + 1:7d} {r[1]:10.4f} {r[2]:9.4f} {r[4]:5.2f} {t[i]:13.6f} {r[3]:6.1f}"
                     f" {r[0]:%Y %m %d}\n")
    print(f"wrote {out}: {len(rows)} events")
    yrs = np.array([r[0].year for r in rows])
    print("events per year:", {y: int(np.sum(yrs == y)) for y in range(y0, y1 + 1)})
    print("first event:", rows[0][0], " Mw range", mw.min(), mw.max())
    kind = np.array([s.split()[0] if " " in s else ("Mw" if s.upper() == s or s == "Mw" else s)
                     for s in ag])
    print("magnitude source:", {a: int(np.sum(ag == a)) for a in np.unique(ag) if np.sum(ag == a) > 20})
    print("by scale:", {k: int(np.sum(kind == k)) for k in np.unique(kind)})
    for m in (4.0, 4.5, 5.0, 5.5, 6.0):
        print(f"  Mw >= {m}: {np.sum(mw >= m)}")
    if "--plot" in args:
        plot(lon, lat, mw, dep, tag, SOURCES[source][3], y0, y1)


def compare(y0, y1):
    """Number of events above each magnitude (log), and events per year, for each source with
    data in the window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = ROOT / "figures" / "indonesia" / f"compare_{y0}-{y1}.png"
    if path.exists():
        print(f"exists, not overwritten: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in SOURCES:
        rows = load(s, y0, y1)
        if not rows:
            continue
        m = np.sort([r[4] for r in rows])
        a1.step(m, len(m) - np.arange(len(m)), where="post", label=f"{SOURCES[s][3]} ({len(m)})")
        yrs = np.array([r[0].year for r in rows])
        a2.plot(range(y0, y1 + 1), [np.sum(yrs == y) for y in range(y0, y1 + 1)], "o-", ms=3)
        print(s, len(m))
    a1.set_yscale("log")
    a1.set_xlabel("magnitude")
    a1.set_ylabel("events with magnitude >= M")
    a1.legend(fontsize=7)
    a2.set_xlabel("year")
    a2.set_ylabel("events per year")
    fig.suptitle(f"Indonesia box, depth <= 50 km, M >= 4, {y0}-{y1}", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


def plot(lon, lat, mw, dep, tag, what, y0, y1):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = ROOT / "figures" / "indonesia" / f"events_{tag}.png"
    if path.exists():
        print(f"exists, not overwritten: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([BOX[0] - 1, BOX[1] + 1, BOX[2] - 1, BOX[3] + 1], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee")
    ax.coastlines("50m", lw=0.6)
    o = np.argsort(mw)                  # big events on top
    sc = ax.scatter(lon[o], lat[o], s=(0.5 if len(mw) > 5000 else 2) * 3.0 ** (mw[o] - 4), c=dep[o],
                    cmap="viridis_r", vmin=0, vmax=50, lw=0, alpha=0.7,
                    transform=ccrs.PlateCarree())
    ax.plot([BOX[0], BOX[1], BOX[1], BOX[0], BOX[0]], [BOX[2], BOX[2], BOX[3], BOX[3], BOX[2]],
            "k--", lw=0.7, transform=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray")
    gl.top_labels = gl.right_labels = False
    fig.colorbar(sc, ax=ax, shrink=0.7, label="depth (km)")
    ax.set_title(f"{what} >= 4, depth <= 50 km, {y0}-{y1}: {len(mw)} events "
                 "(size ~ M; dashed: query box)", fontsize=9)
    fig.savefig(path, dpi=200)          # bbox_inches="tight" drops the map (cartopy 0.25)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
