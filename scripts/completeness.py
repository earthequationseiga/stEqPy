"""Magnitude of completeness Mc of a catalogue (steqpy / Zhuang format), three standard methods
(Woessner & Wiemer 2005, BSSA 95, 684-698), and Mc over time.

- MAXC: the most populated magnitude bin of the non-cumulative distribution, + 0.2 (MAXC
  underestimates Mc on a gradual roll-off; +0.2 is Woessner & Wiemer's correction).
- GFT (Wiemer & Wyss 2000): smallest Mc whose Gutenberg-Richter fit (b by Aki-Utsu above Mc)
  explains the observed counts above Mc to R >= 95% (else 90%); R = 100 - 100 sum|obs - fit| / N.
- MBS (Cao & Gao 2002, Woessner & Wiemer 2005 version): smallest Mc whose b is stable: the mean
  of b(Mc .. Mc + 0.5) within b's Shi & Bolt (1982) uncertainty of b(Mc).
b by Aki (1965) / Utsu: b = log10(e) / (mean(M) - (Mc - dM/2)), dM = bin width.

    python -I scripts/completeness.py CATALOG [--from YEAR] [--dm 0.1] [--plot NAME]
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# The methods live in the package (steqpy.completeness, EventCatalog.completeness); this script
# adds the console table and the figure.
sys.path.insert(0, str(ROOT / "src"))
from steqpy.completeness import b_value, gft_table, maxc, mbs  # noqa: E402


def main():
    args = sys.argv[1:]
    path = Path(args[0])
    dm = float(args[args.index("--dm") + 1]) if "--dm" in args else 0.1
    d = np.loadtxt(path, skiprows=1, ndmin=2)
    year, m = d[:, 6], np.round(d[:, 3], 6)
    y0 = int(args[args.index("--from") + 1]) if "--from" in args else int(year.min())
    keep = year >= y0
    m, year = m[keep], year[keep]
    print(f"{path.name}, {int(year.min())}-{int(year.max())}: {m.size} events, "
          f"M {m.min():.1f}-{m.max():.1f}")
    res = {"MAXC+0.2": maxc(m, dm)}
    table = gft_table(m, dm)
    if "--verbose" in args:
        print("   GFT R:", " ".join(f"{mc:.1f}:{r:.0f}" for mc, r in table))
    lev = next((lv for lv in (95, 90) if any(r >= lv for _, r in table)), None)
    res[f"GFT{lev}"] = next((mc for mc, r in table if r >= lev), math.nan) if lev else math.nan
    res["MBS"] = mbs(m, dm)
    for k, mc in res.items():
        b, se, n = b_value(m, mc, dm)
        print(f"  {k:9s} Mc = {mc:.1f}   b = {b:.3f} +- {se:.3f}   N(>= Mc) = {n}")
    # Mc over time: MAXC+0.2 in consecutive windows of equal event count
    w = 300
    o = np.argsort(year, kind="stable")
    print(f"  Mc(t), MAXC+0.2 in windows of {w} events:")
    rows = []
    for s in range(0, m.size - w // 2, w):
        sel = o[s:s + w]
        rows.append((year[sel].min(), year[sel].max(), maxc(m[sel], dm)))
        print(f"    {int(rows[-1][0])}-{int(rows[-1][1])}: {rows[-1][2]:.1f}")
    if "--plot" in args:
        plot(m, year, dm, res, rows, args[args.index("--plot") + 1], path.name)


def plot(m, year, dm, res, rows, name, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "figures" / name
    if out.exists():
        print(f"exists, not overwritten: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    bins = np.round(np.arange(m.min(), m.max() + dm / 2, dm), 6)
    n = np.array([np.sum(np.abs(m - x) < dm / 2) for x in bins])
    cum = n[::-1].cumsum()[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.semilogy(bins, cum, "s", ms=3, color="k", label="cumulative")
    a1.semilogy(bins, np.where(n > 0, n, np.nan), "^", ms=3, color="gray", label="per bin")
    cols = ("#c44e52", "#4c72b0", "#55a868")
    for (k, mc), col in zip(res.items(), cols):
        b, _, nn = b_value(m, mc, dm)
        x = bins[bins >= mc - 1e-9]
        a1.semilogy(x, nn * 10 ** (-b * (x - mc)), color=col, lw=1,
                    label=f"{k}: Mc {mc:.1f}, b {b:.2f}")
        a1.axvline(mc, color=col, ls=":", lw=0.8)
    a1.set_xlabel("Mw")
    a1.set_ylabel("events")
    a1.legend(fontsize=7)
    a1.set_title(title, fontsize=8)
    for y0, y1, mc in rows:
        a2.plot([y0, y1], [mc, mc], color="k")
    a2.set_xlabel("year")
    a2.set_ylabel("Mc (MAXC + 0.2)")
    a2.set_title("Mc over time, windows of 300 events", fontsize=8)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
