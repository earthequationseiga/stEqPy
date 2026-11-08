"""Figures of a fit, the same views as the website. Used through DeclusteringResult:

    result.plot_events("aftershock")      # map: certain background / uncertain / aftershock
    result.plot_background_rate()         # map: background rate, log scale
    result.plot_histogram()               # P(aftershock) of the target events
    fig = result.plot_summary()           # all of the above on one figure
    events.plot_completeness("cv")        # EventCatalog: events and Mc over time (choose mag_min)

Each returns the matplotlib Axes (plot_summary: the Figure) and accepts ax= to draw into your
own. matplotlib is only needed here (optional dependency). Maps are plain lon/lat axes with the
study polygon; for coastlines pass a cartopy GeoAxes (e.g. projection=ccrs.PlateCarree()) as
ax, then call ax.coastlines(): the data are drawn with transform=PlateCarree.
Classes by P(aftershock) = 1 - phi (result.BAND_LO, BAND_HI), colours from Crameri's Oslo
(dark blue = aftershock); background rate in Crameri's Bilbao (dark red = high), as the site.
"""
import numpy as np

from .result import BAND_HI, BAND_LO, DEG_KM

C_AFT, C_UNC, C_BG, C_OUT = "#15395b", "#6388c6", "#d8dade", "#c6c8c6"
BILBAO = ["#ffffff", "#eaeaea", "#d7d7d3", "#cac7bc", "#c2bca6", "#bbb291", "#b5a57c", "#ae956d", "#aa8665",
          "#a67960", "#a26b59", "#9d5c52", "#934b47", "#843938", "#732726", "#601516", "#4d0001"]
KINDS = ("all", "background", "uncertain", "aftershock")


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plotting needs matplotlib (pip install matplotlib)") from None
    return plt


def _geo_kw(ax):
    """transform=PlateCarree for a cartopy GeoAxes, nothing for plain axes."""
    if hasattr(ax, "coastlines"):
        import cartopy.crs as ccrs
        return {"transform": ccrs.PlateCarree()}
    return {}


def _ax(ax, figsize=(6, 5)):
    return ax if ax is not None else _plt().subplots(figsize=figsize)[1]


def _lonlat(result):
    c, r = result.catalog, result.region
    if result.projection is None:
        return c.x, c.y, r.polygon_x, r.polygon_y
    lon, lat = result.projection.inverse(c.x, c.y)
    plon, plat = result.projection.inverse(r.polygon_x, r.polygon_y)
    return lon, lat, plon, plat


def _frame(ax, result, plon, plat, pad=0.15):
    kw = _geo_kw(ax)
    ax.plot(plon, plat, color="#26323d", lw=1.2, **kw)
    w, e, s, n = plon.min(), plon.max(), plat.min(), plat.max()
    dx, dy = (e - w) * pad, (n - s) * pad
    if kw:
        ax.set_extent([w - dx, e + dx, s - dy, n + dy], crs=kw["transform"])
    else:
        ax.set_xlim(w - dx, e + dx)
        ax.set_ylim(s - dy, n + dy)
        ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))   # degrees roughly to scale
        ax.set_xlabel("longitude (°)")
        ax.set_ylabel("latitude (°)")


def _mag_cutoff(result):
    return float(((result.config or {}).get("input") or {}).get("mag_min", 0.0))


def events(result, kind="all", ax=None, show_outside=True):
    """Map of the events coloured by class: kind "all", "background" (P(aftershock) < BAND_LO),
    "uncertain" or "aftershock" (> BAND_HI). Events outside the study (polygon or window) are
    grey when show_outside; marker area grows with magnitude, largest drawn on top."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    ax = _ax(ax)
    kw = _geo_kw(ax)
    lon, lat, plon, plat = _lonlat(result)
    pa = 1.0 - result.final_background_prob
    target = result.ind == 1
    size = 3.0 * 1.8 ** np.asarray(result.catalog.m)            # m = magnitude above the cutoff
    cls = np.where(pa > BAND_HI, 2, np.where(pa < BAND_LO, 0, 1))
    if show_outside:
        o = ~target
        ax.scatter(lon[o], lat[o], s=size[o], c=C_OUT, alpha=0.5, lw=0, label="outside the study", **kw)
    labels = {0: f"background (P < {BAND_LO:g})", 1: "uncertain", 2: f"aftershock (P > {BAND_HI:g})"}
    for k, col in ((0, C_BG), (1, C_UNC), (2, C_AFT)):
        if kind != "all" and KINDS.index(kind) - 1 != k:
            continue
        sel = target & (cls == k)
        o = np.argsort(size[sel])
        ax.scatter(lon[sel][o], lat[sel][o], s=size[sel][o], c=col, edgecolors="#26323d",
                   linewidths=0.3, label=f"{labels[k]}: {int(sel.sum())}", **kw)
    _frame(ax, result, plon, plat)
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    ax.set_title(f"{'Events' if kind == 'all' else kind.capitalize() + ' events'}, "
                 f"M ≥ {_mag_cutoff(result):g}", fontsize=10)
    return ax


def background_rate(result, ax=None, n=250, decades=4):
    """Map of the background rate mu * u(x) on an n x n lon/lat grid over the polygon's box, in
    events (M >= cutoff) per year per 10,000 km^2, log colour scale over at most `decades`."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    ax = _ax(ax)
    kw = _geo_kw(ax)
    _, _, plon, plat = _lonlat(result)
    w, e, s, nn = plon.min(), plon.max(), plat.min(), plat.max()
    dx, dy = (e - w) * 0.15, (nn - s) * 0.15
    glon = np.linspace(w - dx, e + dx, n)
    glat = np.linspace(s - dy, nn + dy, n)
    LON, LAT = np.meshgrid(glon, glat)
    u = result.rates_lonlat(LON.ravel(), LAT.ravel())[:, 0]       # column 0: u (without mu)
    rate = (result.params.mu * u * 365.25 * 1e4 / DEG_KM ** 2).reshape(LON.shape)
    hi = float(rate.max())
    lo = max(float(np.percentile(rate[rate > 0], 5)), hi * 10.0 ** -decades)
    pm = ax.pcolormesh(LON, LAT, np.clip(rate, lo, hi), shading="auto",
                       cmap=LinearSegmentedColormap.from_list("bilbao", BILBAO),
                       norm=LogNorm(lo, hi), **kw)
    _plt().colorbar(pm, ax=ax, shrink=0.8,
                    label=f"M ≥ {_mag_cutoff(result):g} background events / yr / 10⁴ km²")
    _frame(ax, result, plon, plat)
    ax.set_title("Background rate", fontsize=10)
    return ax


def histogram(result, ax=None):
    """P(aftershock) of the target events in result.summary()'s bins, coloured by band."""
    ax = _ax(ax, (6, 3.5))
    h = result.summary()
    s = h["histogram_p_aftershock"]
    edges, counts = np.asarray(s["edges"]), np.asarray(s["counts"])
    mid = (edges[:-1] + edges[1:]) / 2
    cols = [C_BG if m < BAND_LO else C_AFT if m > BAND_HI else C_UNC for m in mid]
    ax.bar(mid, counts, width=np.diff(edges) * 0.95, color=cols, edgecolor="#26323d", lw=0.3)
    for x in (BAND_LO, BAND_HI):
        ax.axvline(x, color="#55616c", ls=":", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(aftershock)")
    ax.set_ylabel("target events")
    ax.set_title(f"{h['n_targets']} targets: {h['certain_background']} background, "
                 f"{h['uncertain']} uncertain, {h['certain_aftershock']} aftershocks", fontsize=10)
    return ax


def summary(result, figsize=(13, 10)):
    """One figure: events by class, background rate, histogram, parameters."""
    plt = _plt()
    fig = plt.figure(figsize=figsize)
    events(result, "all", fig.add_subplot(2, 2, 1))
    background_rate(result, fig.add_subplot(2, 2, 2))
    histogram(result, fig.add_subplot(2, 2, 3))
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    p = result.params
    rows = [f"{k:6s} {v:.4g}" for k, v in vars(p).items()]
    if result.K is not None and np.isnan(p.A):
        rows.append(f"K      {result.K:.4g}  (A undefined)")
    rows += ["", f"-log L {result.neg_loglik:.2f}", f"converged {result.converged}"]
    rows += [f"warning: {w}" for w in (result.checks or {}).get("warnings", [])]
    rows += [f"FAILED: {f}" for f in (result.checks or {}).get("failed", [])]
    ax.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=9, wrap=True)
    ax.set_title("Fitted parameters (c in days, D in deg²)", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def completeness(events, method="cv", step=365.25 / 4, window=365.25 / 2, ax=None):
    """Events per step (bars, log scale) and Mc over time (dots, right axis) of an EventCatalog,
    as the website draws on upload. Dates on the x axis when the catalogue has t0."""
    ax = _ax(ax, (8, 3.5))
    s = events.mc_over_time(method, step, window)
    to_x = (lambda d: events.t0 + (np.asarray(d) * 86_400_000).astype("timedelta64[ms]"))         if events.t0 is not None else (lambda d: np.asarray(d))
    width = (to_x(s["bin_start"] + step) - to_x(s["bin_start"])) * 0.95
    ax.bar(to_x(s["bin_start"]), np.maximum(s["counts"], 0), width=width, align="edge",
           color="#cfc8b8", log=True)
    span = lambda d: f"{d / (365.25 / 12):.3g} months" if d >= 28 else f"{d:.3g} days"
    ax.set_ylabel(f"events / {span(step)}")
    ax.set_xlabel("time" if events.t0 is not None else "days")
    ax2 = ax.twinx()
    ok = np.isfinite(s["mc"])
    ax2.plot(to_x(s["t"][ok]), s["mc"][ok], "o", ms=3, color="#5a55c8")
    ax2.set_ylabel(f"Mc ({method})", color="#5a55c8")
    ax.set_title(f"Events and completeness: Mc every {span(step)} from the {span(window)} before "
                 f"(no dot: too few events)", fontsize=9)
    return ax
