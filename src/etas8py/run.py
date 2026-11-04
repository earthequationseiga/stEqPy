"""Driver: port of the main program, readata, dav, outrates and outprob (etas8p.f, etas8out.f).

Usage:  python -m etas8py.run data/jma/jap.in --out results/python_run [--schedule full|short]

Differences from the Fortran (all I/O, none numerical):
- the per-pass dump of every zprob to stdout is not printed;
- pmatr.dat (all event pairs, O(N^2) disk) is not written.
"""
import argparse
import math
import shlex
import time
from pathlib import Path

import numpy as np

from . import kernels as K
from .davidn import davidn

DEG = 3.1415926535 / 180.0  # the original's degree->radian constant


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


def ycentroid(tx, ty):
    npoly = len(tx)
    area = 0.0
    ymoment = 0.0
    for i in range(npoly):
        i1 = (i + 1) % npoly
        temp = tx[i] * ty[i1] - tx[i1] * ty[i]
        area = area + temp
        ymoment = ymoment + (ty[i] + ty[i1]) * temp
    area = area / 2
    return ymoment / 6.0 / area


def _sgn(v):
    return 1 if v > 0.0 else (-1 if v < 0.0 else 0)


def polyse(px, py, xx, yy):
    """Point in polygon (crossing count): 1 inside, -1 outside, 0 on the boundary.
    px, py are closed (last vertex == first)."""
    n = len(px) - 1
    flag = np.empty(len(xx), dtype=np.int64)
    for k in range(len(xx)):
        x, y = xx[k], yy[k]
        ip = 0
        te = None
        for i in range(n):
            is_ = _sgn(y - py[i]) * _sgn(py[i + 1] - y)
            if is_ < 0:
                continue
            if is_ > 0:
                temp = (px[i + 1] * abs(y - py[i]) + px[i] * abs(py[i + 1] - y)
                        - abs(py[i + 1] - py[i]) * x)
                if temp > 0:
                    ip += 1
                elif temp == 0:
                    te = 0
                    break
            elif y == py[i] and py[i + 1] > y:
                if x - px[i] < 0:
                    ip += 1
                elif x - px[i] == 0:
                    te = 0
                    break
            elif y == py[i + 1] and py[i] > y:
                if x - px[i + 1] < 0:
                    ip += 1
                elif x - px[i + 1] == 0:
                    te = 0
                    break
            elif y == py[i] and py[i + 1] < y:
                if x - px[i] == 0:
                    te = 0
                    break
            elif y == py[i + 1] and py[i] < y:
                if x - px[i + 1] == 0:
                    te = 0
                    break
            elif y == py[i + 1] and py[i] == y:
                if _sgn(x - px[i]) * _sgn(px[i + 1] - x) > 0:
                    te = 0
                    break
        flag[k] = te if te is not None else (1 if ip % 2 == 1 else -1)
    return flag


def readata(infile):
    """Parse the control file + catalog exactly like readata. Returns a dict of arrays."""
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
    tx = np.append(poly[:, 0], poly[0, 0])
    ty = np.append(poly[:, 1], poly[0, 1])
    ymean = ycentroid(poly[:, 0], poly[:, 1])
    cosm = math.cos(ymean * DEG)
    tx = tx * cosm                              # flat local coords: shrink longitude
    mx, my, rtx1, rtx2, rty1, rty2 = r.read(6)
    grid = (int(mx), int(my), float(rtx1) * cosm, float(rtx2) * cosm, float(rty1), float(rty2))
    b = np.array(list(map(float, r.read(8))))   # (mu, A, c, alfa, p, d, q, gamma)
    npp, bwm = r.read(2)
    ioptimise = int(r.read(1)[0])

    # catalog: header line, then id lon lat mag t ... ; same filters/stop rule as Fortran
    cat_path = infile.parent / hypodata
    rows = np.loadtxt(cat_path, skiprows=1, usecols=(1, 2, 3, 4))
    mag_cut = xmg0 - float(np.float32(0.000001))
    keep = []
    last_t = None
    for lon, lat, mag, t in rows:
        if t > tz + zmin:
            break
        if last_t is not None and t < last_t:
            raise ValueError(f"Reverse data: {last_t} {t}")
        if mag > mag_cut and t > zmin:
            keep.append((lon * cosm, lat, mag, t))
            last_t = t
        # (Fortran compares against the last *kept* event too: rejected rows are overwritten)
    xx, yy, zmg, zz = (np.array(c) for c in zip(*keep))
    ind = polyse(tx, ty, xx, yy)
    ind[(zz > tz + zmin) | (zz < tstart)] = 0
    zmg = zmg - xmg0
    zz = zz - zmin
    return dict(xx=xx, yy=yy, zz=zz, zmg=zmg, ind=ind, tx=tx, ty=ty, npoly=npoly,
                tz=tz, tstart=tstart - zmin, grid=grid, b=b, npp=int(npp),
                bwm=float(bwm), ioptimise=ioptimise)


class Model:
    """Holds the state the Fortran keeps in COMMON blocks."""

    def __init__(self, d):
        self.__dict__.update(d)
        self.nn = len(self.xx)
        self.x = np.sqrt(self.b)
        self.zprob = np.ones(self.nn)
        self.zbkgd = np.zeros(self.nn)
        self.zbandw = None
        self.xint0 = 0.0
        self.nfunc = 0
        self.fast_xint = False
        self.parallel = False                   # use the prange kernels (same results)

    def bkgdcalc(self):
        fn = K.bkgdcalc_par if self.parallel else K.bkgdcalc
        self.xint0 = fn(self.xx, self.yy, self.zprob, self.zbandw, self.tx, self.ty,
                        self.npoly, self.tz, self.tstart, self.zbkgd)

    def probcalc(self):
        fn = K.probcalc_par if self.parallel else K.probcalc
        fn(self.x, self.zbkgd, self.xx, self.yy, self.zz, self.zmg, self.zprob)

    def func15(self, b, log=print):
        grad = np.empty(8)
        fn = K.func15_par if self.parallel else K.func15
        f = fn(np.asarray(b, dtype=np.float64), self.ind, self.zbkgd, self.xint0,
               self.xx, self.yy, self.zz, self.zmg, self.tx, self.ty, self.npoly,
               self.tz, self.tstart, grad, self.fast_xint)
        self.nfunc += 1
        log(f" function value= {f:12.4f}\n Gradient= " + "".join(f"{v:12.4f}" for v in grad))
        return f, list(grad)

    def dav(self, out, log):
        x = list(np.sqrt(self.b))
        davidn(x, self.func15, log)
        self.x = np.array(x)
        self.b = self.x ** 2
        log(" mle = " + " ".join(f"{v:.5E}" for v in self.b))
        with open(out / "para", "a") as fh:
            fh.write(" " + "".join(f"{v:20.9f}" for v in self.b) + "\n")

    def outrates(self, path):
        mx, my, tx1, tx2, ty1, ty2 = self.grid
        gx = np.array([tx1 + (tx2 - tx1) / (mx - 1) * i for i in range(mx)])
        gy = np.array([ty1 + (ty2 - ty1) / (my - 1) * j for j in range(my)])
        fn = K.outrates_grid_par if self.parallel else K.outrates_grid
        rates = fn(self.x, self.xx, self.yy, self.zz, self.zmg, self.zprob,
                   self.zbandw, self.tz, self.tstart, gx, gy)
        with open(path, "w") as fh:
            fh.write(f"{mx:12d}\n{my:12d}\n")
            np.savetxt(fh, rates, fmt="%20.10f")

    def outprob(self, path):
        with open(path, "w") as fh:
            for i in range(self.nn):
                fh.write(f" {i + 1:8d}{self.zprob[i]:20.10f}{self.zbandw[i]:20.10f}"
                         f"{self.zbkgd[i]:20.10f}\n")


SCHEDULES = {
    # outer iterations of (inner bkgd/prob passes, then MLE)
    "full": [12 - k for k in range(1, 12)],     # the original: 11+10+...+1 = 66 passes
    "short": [2],                               # quick smoke test: 2 passes + 1 MLE
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default="results/python_run")
    ap.add_argument("--schedule", choices=SCHEDULES, default="full")
    ap.add_argument("--fast-xint", action="store_true",
                    help="fused polygon integrals in the likelihood (same results to rounding)")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads for bkgdcalc/probcalc/func15; 0 = serial 1:1 port loops")
    a = ap.parse_args(argv)
    if a.threads > 0:
        import numba
        numba.set_num_threads(a.threads)
        # small chunks balance the O(i) per-event cost of xlamb across threads
        numba.set_parallel_chunksize(16)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "para").write_text("")
    logf = open(out / "run.log", "w")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    timings = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        r = fn()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return r

    m = Model(timed("readata", lambda: readata(a.infile)))
    m.fast_xint = a.fast_xint
    m.parallel = a.threads > 0
    log(f" {int((m.ind == 1).sum())} of {m.nn} events selected.")
    m.zbandw = timed("bandwcalc", lambda: K.bandwcalc(m.xx, m.yy, m.npp, m.bwm))
    for k, reps in enumerate(SCHEDULES[a.schedule], 1):
        log(f" Iteration {k}")
        for _ in range(reps):
            timed("bkgdcalc", m.bkgdcalc)
            timed("probcalc", m.probcalc)
        if m.ioptimise == 1:
            timed("dav", lambda: m.dav(out, log))
    timed("outrates", lambda: m.outrates(out / "rates.dat"))
    timed("outprob", lambda: m.outprob(out / "probs.dat"))

    log(f" func15 calls: {m.nfunc}")
    for k, v in timings.items():
        log(f" time {k:10s} {v:10.3f} s")
        print(f"{k:10s} {v:10.3f} s")
    print(f"func15 calls: {m.nfunc}")
    logf.close()


if __name__ == "__main__":
    main()
