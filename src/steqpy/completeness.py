"""Magnitude of completeness Mc: the smallest magnitude above which (nearly) every event is
recorded. Below Mc the catalogue misses events, and a declustering fit reads the gaps as
changes in seismicity (e.g. p < 1 when small aftershocks go missing after big events), so fit
at or above Mc. Used through EventCatalog.completeness / .b_value; plain functions here.

Methods (Woessner & Wiemer 2005, BSSA 95, 684-698, for the first three):
- "maxc": the most populated magnitude bin, + 0.2 (MAXC alone underestimates Mc on a gradual
  roll-off; +0.2 is Woessner & Wiemer's correction).
- "gft" (Wiemer & Wyss 2000): smallest Mc whose Gutenberg-Richter fit explains the counts above
  it to R >= 95% (else 90%); R = 100 - 100 sum|obs - fit| / N.
- "mbs" (Cao & Gao 2002, Woessner & Wiemer version): smallest Mc whose b is stable: the mean of
  b(Mc .. Mc + 0.5) within b's Shi & Bolt (1982) uncertainty of b(Mc).
- "cv" (Godano, Petrillo & Lippiello 2023, GJI 236, 994-1001): above Mc, magnitudes are
  exponential, whose coefficient of variation is 1. Our automated rule (the paper reads the
  plateau by eye): smallest m with n >= 50 and CV >= 1 - max(0.05, 1/sqrt(n)). Some complete
  catalogues plateau below that (JMA at about 0.93, as the paper reports for Japan): then NaN.
b by Aki (1965) / Utsu: b = log10(e) / (mean(M) - (Mc - dM/2)), dM = magnitude bin width.
All return NaN when there are fewer than NMIN events to judge.
"""
import math

import numpy as np

NMIN = 50
METHODS = ("maxc", "gft", "mbs", "cv")


def bin_width(m):
    """Finest of 1, 0.1, 0.01 that the magnitudes use (as the website does)."""
    for dm in (1.0, 0.1):
        if np.all(np.abs(m / dm - np.round(m / dm)) < 1e-6):
            return dm
    return 0.01


def b_value(m, mc, dm):
    """Aki-Utsu b of magnitudes >= mc, its Shi & Bolt standard error, and their count."""
    x = m[m >= mc - 1e-9]
    if x.size < 2:
        return math.nan, math.nan, x.size
    b = math.log10(math.e) / (x.mean() - (mc - dm / 2))
    se = 2.3 * b * b * math.sqrt(np.sum((x - x.mean()) ** 2) / (x.size * (x.size - 1)))
    return b, se, x.size


def maxc(m, dm, correction=0.2):
    if m.size < NMIN:
        return math.nan
    edges = np.arange(m.min() - dm / 2, m.max() + dm, dm)
    n, _ = np.histogram(m, edges)
    return float(round(edges[np.argmax(n)] + dm / 2 + correction, 6))


def gft_table(m, dm, nmin=NMIN):
    """(Mc, R) for each candidate Mc with at least nmin events above it."""
    table = []
    for mc in np.round(np.arange(m.min(), m.max() - 1, dm), 6):
        b, _, n = b_value(m, mc, dm)
        if n < nmin:
            break
        a = math.log10(n) + b * (mc - dm / 2)           # bin mc starts at mc - dm/2: N there = n
        bins = np.round(np.arange(mc, m.max() + dm / 2, dm), 6)
        obs = np.array([np.sum(np.abs(m - x) < dm / 2) for x in bins])
        cum = 10 ** (a - b * (bins - dm / 2))           # N(>= x - dm/2), first one = n
        fit = cum - np.append(cum[1:], 0.0)
        table.append((float(mc), 100 - 100 * np.sum(np.abs(obs - fit)) / n))
    return table


def gft(m, dm, levels=(95, 90)):
    table = gft_table(m, dm)
    for lev in levels:
        for mc, r in table:
            if r >= lev:
                return mc
    return math.nan


def mbs(m, dm, width=0.5, nmin=NMIN):
    for mc in np.round(np.arange(m.min(), m.max() - width, dm), 6):
        b, se, n = b_value(m, mc, dm)
        if n < nmin:
            break
        bs = [b_value(m, mc + k * dm, dm)[0] for k in range(int(round(width / dm)) + 1)]
        if abs(np.mean(bs) - b) <= se:
            return float(mc)
    return math.nan


def cv(m, dm, nmin=NMIN):
    m = np.sort(m)
    mth = round(m[0] / dm) * dm if m.size else math.nan
    while m.size >= nmin and mth < m[-1]:
        x = m[m >= mth - 1e-9] - (mth - dm / 2)
        if x.size < nmin:
            break
        if x.std(ddof=1) / x.mean() >= 1 - max(0.05, 1 / math.sqrt(x.size)):
            return round(mth, 6)
        mth = round((mth + dm) / dm) * dm
    return math.nan


def mc(m, method="maxc", dm=None):
    """Mc of magnitudes m by one of METHODS; dm: bin width (default: bin_width(m))."""
    m = np.round(np.asarray(m, dtype=float), 6)
    dm = bin_width(m) if dm is None else dm
    fn = {"maxc": maxc, "gft": gft, "mbs": mbs, "cv": cv}.get(method)
    if fn is None:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, not {method!r}")
    return fn(m, dm)
