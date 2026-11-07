"""Numba ports of the numerical routines in etas8fun.f, frint.f and etas8p.f.

First-pass translation: same loops, same operation order, same (wasteful) work as the
Fortran, so speed and results are directly comparable. Indices are 0-based; event i's
"earlier events" are j < i (catalog is time-sorted).

Parameters arrive as x = sqrt(theta), theta = (mu, A, c, alpha, p, D, q, gamma), because the
optimiser works on square roots to keep every parameter positive.
"""
import math
from functools import lru_cache

import numpy as np
from numba import njit, prange

PI = 3.14159265358979323846264338327
# Fortran single-precision literals (1e-10, 1e-20) promoted to double, kept bit-identical.
E10 = float(np.float32(1e-10))
E20 = float(np.float32(1e-20))
NDIV = 1000  # sub-segments per polygon edge in polyint

# Radial functions integrated over the polygon (frint's `func` argument).
PGAUSS, FR, DD_FR, DQ_FR, DGAMMA_FR = 0, 1, 2, 3, 4


@njit(cache=True)
def radial(kind, r, w):
    """F(r) = int_0^r f(s) s ds for the kernel f selected by `kind` (and its parameter
    derivatives). w = (bandwidth,) for PGAUSS, (gamma, D, q, mag) for the ETAS kernels."""
    if kind == PGAUSS:
        return 1.0 / 2.0 / PI * (1.0 - math.exp(-(r * r) / 2.0 / (w[0] * w[0])))
    gamma, d, q, xmag = w[0], w[1], w[2], w[3]
    ssig = d * math.exp(gamma * xmag)
    if kind == FR:
        return (1.0 - (1 + r * r / ssig) ** (1.0 - q)) / PI / 2.0
    if kind == DD_FR:
        return (-q + 1) * (1 + r * r / ssig) ** (-q) / d * r * r / ssig / 2.0 / PI
    if kind == DQ_FR:
        return (1.0 + r * r / ssig) ** (-q + 1) * math.log(1.0 + r * r / ssig) / PI / 2.0
    # DGAMMA_FR
    return (-q + 1) * (1 + r * r / ssig) ** (-q) * xmag * r * r / ssig / 2.0 / PI


@njit(cache=True)
def dgauss(r, w0):
    return 1.0 / 2.0 / PI / (w0 * w0) * math.exp(-(r * r) / 2.0 / (w0 * w0))


@njit(cache=True)
def frint(kind, w, x1, y1, x2, y2, xc, yc, eps=E10, eps_r=E20):
    """Integral of the radial kernel centred at (xc, yc) over the triangle
    (xc,yc)-(x1,y1)-(x2,y2): Simpson's rule in angle on F(r), with the middle point taken
    where the angle bisector hits the segment. Sign follows the triangle orientation.
    eps: skip triangles with |2*area| < eps, and clamp cos(angle) to 1-eps; eps_r: skip
    pieces whose endpoints both sit on the centre (original 1e-10, 1e-20)."""
    det = (x1 * y2 + y1 * xc + x2 * yc) - (x1 * yc + x2 * y1 + xc * y2)
    sgn = -1 if det < 0.0 else 1
    if abs(det) < eps:
        return 0.0
    r1 = math.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
    r2 = math.sqrt((x2 - xc) ** 2 + (y2 - yc) ** 2)
    r12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    theta = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2)
    if abs(theta) > 1.0:
        theta = 1.0 - eps  # original also maps theta < -1 to ~+1 (kept as-is)
    theta = math.acos(theta)
    if r1 + r2 > eps_r:
        x0 = x1 + r1 / (r1 + r2) * (x2 - x1)
        y0 = y1 + r1 / (r1 + r2) * (y2 - y1)
    else:
        return 0.0
    r0 = math.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
    f1 = radial(kind, r1, w)
    f2 = radial(kind, r0, w)
    f3 = radial(kind, r2, w)
    return sgn * (f1 / 6 + f2 * 2.0 / 3.0 + f3 / 6.0) * theta


@njit(cache=True)
def polyint(kind, w, xp, yp, npoly, xc, yc, ndiv=NDIV, eps=E10, eps_r=E20):
    """Integral of the radial kernel over the polygon (Green's theorem: sum of signed
    triangles fanning out from the kernel centre to ndiv pieces of every edge)."""
    s = 0.0
    for j in range(npoly):
        for i in range(1, ndiv + 1):
            x1 = xp[j] + (xp[j + 1] - xp[j]) / ndiv * (i - 1)
            y1 = yp[j] + (yp[j + 1] - yp[j]) / ndiv * (i - 1)
            x2 = xp[j] + (xp[j + 1] - xp[j]) / ndiv * i
            y2 = yp[j] + (yp[j + 1] - yp[j]) / ndiv * i
            s += frint(kind, w, x1, y1, x2, y2, xc, yc, eps, eps_r)
    return s


@njit(cache=True)
def bandwcalc(xx, yy, npp, bwm):
    """Bandwidth per event = distance to its npp-th nearest other event, floored at bwm.
    Brute force O(N^2 * npp) insertion into a sorted buffer, as in the original."""
    nn = xx.shape[0]
    zbandw = np.empty(nn)
    buf = np.empty(npp)
    for i in range(nn):
        buf[:] = 10e39
        for j in range(nn):
            if i == j:
                continue
            temp = math.sqrt((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2)
            for k in range(npp):
                if buf[k] > temp:
                    for kk in range(npp - 1, k, -1):
                        buf[kk] = buf[kk - 1]
                    buf[k] = temp
                    break
        zbandw[i] = max(buf[npp - 1], bwm)
    return zbandw


@njit(cache=True)
def bkgdcalc(xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, ndiv=NDIV, eps=E10,
             eps_r=E20):
    """Background rate at every event (variable-bandwidth Gaussian sum weighted by the
    background probabilities, self term included) -> zbkgd in place.
    Returns xint0 = sum_i zprob_i * integral of kernel i over the polygon."""
    nn = xx.shape[0]
    w = np.empty(1)
    for i in range(nn):
        s = 0.0
        for j in range(nn):
            r0 = math.sqrt((xx[j] - xx[i]) ** 2 + (yy[j] - yy[i]) ** 2)
            s += zprob[j] * dgauss(r0, zbandw[j])
        zbkgd[i] = s / (tz - tstart)
    s = 0.0
    for i in range(nn):
        w[0] = zbandw[i]
        s += zprob[i] * polyint(PGAUSS, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps, eps_r)
    return s


@njit(cache=True)
def xlamb(i, b, zbkgd, xx, yy, zz, zmg, g1):
    """Conditional intensity lambda at event i (returned) and its gradient w.r.t. the
    sqrt-parameters b (written to g1). O(i) sum over earlier events."""
    xmu = b[0] ** 2
    a2 = b[1] ** 2
    c = b[2] ** 2
    alfa = b[3] ** 2
    p = b[4] ** 2
    d = b[5] ** 2
    q = b[6] ** 2
    gamma = b[7] ** 2

    s = xmu * zbkgd[i]
    sg1 = zbkgd[i]
    sg2 = sg3 = sg4 = sg5 = sg6 = sg7 = sg8 = 0.0
    for j in range(i):
        delt = zz[i] - zz[j]
        pr1 = math.exp(alfa * zmg[j])
        pr2 = (p - 1) / c * (1.0 + delt / c) ** (-p)
        ssig = d * math.exp(gamma * zmg[j])
        bbb = (q - 1) / ssig / PI
        dist2 = (xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2
        pr3 = bbb * (dist2 / ssig + 1.0) ** (-q)

        s += a2 * pr1 * pr2 * pr3
        sg2 += pr1 * pr2 * pr3
        pr2_c = pr2 * (-1.0 / c - p / (c + delt) + p / c)
        sg3 += a2 * pr1 * pr2_c * pr3
        pr1_alfa = pr1 * zmg[j]
        sg4 += a2 * pr1_alfa * pr2 * pr3
        pr2_p = pr2 * (1.0 / (p - 1) - math.log(1.0 + delt / c))
        sg5 += a2 * pr1 * pr2_p * pr3
        pr3_d = pr3 / d * (-1.0 + q * (1.0 - 1.0 / (1.0 + dist2 / ssig)))
        sg6 += a2 * pr1 * pr2 * pr3_d
        pr3_q = pr3 * (1.0 / (q - 1) - math.log(1 + dist2 / ssig))
        sg7 += a2 * pr1 * pr2 * pr3_q
        pr3_gamma = pr3 * (-zmg[j] + q * zmg[j] * (1.0 - 1.0 / (1.0 + dist2 / ssig)))
        sg8 += a2 * pr1 * pr2 * pr3_gamma

    g1[0] = sg1 * 2 * b[0]
    g1[1] = sg2 * 2 * b[1]
    g1[2] = sg3 * 2 * b[2]
    g1[3] = sg4 * 2 * b[3]
    g1[4] = sg5 * 2 * b[4]
    g1[5] = sg6 * 2 * b[5]
    g1[6] = sg7 * 2 * b[6]
    g1[7] = sg8 * 2 * b[7]
    return s


@njit(cache=True)
def xint(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, h, ndiv=NDIV, eps=E10, eps_r=E20):
    """Integral over [tstart, tz] x polygon of the offspring intensity of event i (returned)
    and its gradient (h). Time part closed form, space part numerical (4 polyint calls)."""
    a2 = b[1] ** 2
    c = b[2] ** 2
    alfa = b[3] ** 2
    p = b[4] ** 2
    d = b[5] ** 2
    q = b[6] ** 2
    gamma = b[7] ** 2

    if zz[i] > tstart:
        ttemp = tz - zz[i]
        gi = 1.0 - (1.0 + ttemp / c) ** (1 - p)
        gic = -(1.0 - gi) * (1.0 - p) * (1.0 / (c + ttemp) - 1.0 / c)
        gip = -(1.0 - gi) * (math.log(c) - math.log(c + ttemp))
    else:
        ttemp2 = tz - zz[i]
        ttemp1 = tstart - zz[i]
        gi2 = 1 - (1.0 + ttemp2 / c) ** (1 - p)
        gic2 = -(1.0 - gi2) * (1.0 - p) * (1.0 / (c + ttemp2) - 1.0 / c)
        gip2 = -(1.0 - gi2) * (math.log(c) - math.log(c + ttemp2))
        gi1 = 1 - (1.0 + ttemp1 / c) ** (1 - p)
        gic1 = -(1.0 - gi1) * (1.0 - p) * (1.0 / (c + ttemp1) - 1.0 / c)
        gip1 = -(1.0 - gi1) * (math.log(c) - math.log(c + ttemp1))
        gi = gi2 - gi1
        gic = gic2 - gic1
        gip = gip2 - gip1

    w = np.empty(4)
    w[0] = gamma
    w[1] = d
    w[2] = q
    w[3] = zmg[i]
    si = polyint(FR, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps, eps_r)
    sid = polyint(DD_FR, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps, eps_r)
    siq = polyint(DQ_FR, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps, eps_r)
    sigamma = polyint(DGAMMA_FR, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps, eps_r)

    sk = a2 * math.exp(alfa * zmg[i])
    fv = sk * gi * si
    h[0] = 0.0
    h[1] = sk * gi * si / a2 * 2.0 * b[1]
    h[2] = sk * gic * si * 2 * b[2]
    h[3] = sk * gi * si * zmg[i] * 2 * b[3]
    h[4] = sk * gip * si * 2 * b[4]
    h[5] = sk * gi * sid * 2 * b[5]
    h[6] = sk * gi * siq * 2 * b[6]
    h[7] = sk * gi * sigamma * 2 * b[7]
    return fv


@njit(cache=True)
def _fr4(r, ssig, d, q, xmag):
    """fr, dd_fr, dq_fr, dgamma_fr at one radius, sharing t = 1 + r^2/ssig and t^-q
    (1 pow + 1 log instead of 4 exp + 4 pow + 1 log). Same formulas, regrouped:
    t^(1-q) is computed as t * t^-q, so values differ from radial() by rounding only."""
    u = r * r / ssig
    t = 1.0 + u
    tq = t ** (-q)
    c0 = 1.0 / 2.0 / PI
    return ((1.0 - t * tq) * c0,
            (1.0 - q) * tq / d * u * c0,
            t * tq * math.log(t) * c0,
            (1.0 - q) * tq * xmag * u * c0)


@njit(cache=True)
def polyint_etas4(gamma, d, q, xmag, xp, yp, npoly, xc, yc, ndiv=NDIV, eps=E10, eps_r=E20):
    """The four polyint calls of xint (FR, DD_FR, DQ_FR, DGAMMA_FR) in one pass: segment
    geometry computed once, ssig once per event, and each sub-segment endpoint evaluated
    once (shared by its two neighbouring pieces). Same quadrature as polyint/frint."""
    ssig = d * math.exp(gamma * xmag)
    s0 = s1 = s2 = s3 = 0.0
    for j in range(npoly):
        x1 = xp[j]
        y1 = yp[j]
        r1 = math.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
        a0, a1, a2, a3 = _fr4(r1, ssig, d, q, xmag)
        for i in range(1, ndiv + 1):
            x2 = xp[j] + (xp[j + 1] - xp[j]) / ndiv * i
            y2 = yp[j] + (yp[j + 1] - yp[j]) / ndiv * i
            r2 = math.sqrt((x2 - xc) ** 2 + (y2 - yc) ** 2)
            c0, c1, c2, c3 = _fr4(r2, ssig, d, q, xmag)
            det = (x1 * y2 + y1 * xc + x2 * yc) - (x1 * yc + x2 * y1 + xc * y2)
            if abs(det) >= eps and r1 + r2 > eps_r:
                sgn = -1 if det < 0.0 else 1
                r12 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                theta = (r1 * r1 + r2 * r2 - r12 * r12) / (2.0 * r1 * r2)
                if abs(theta) > 1.0:
                    theta = 1.0 - eps
                theta = math.acos(theta)
                x0 = x1 + r1 / (r1 + r2) * (x2 - x1)
                y0 = y1 + r1 / (r1 + r2) * (y2 - y1)
                r0 = math.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
                b0, b1, b2, b3 = _fr4(r0, ssig, d, q, xmag)
                s0 += sgn * (a0 / 6 + b0 * 2.0 / 3.0 + c0 / 6.0) * theta
                s1 += sgn * (a1 / 6 + b1 * 2.0 / 3.0 + c1 / 6.0) * theta
                s2 += sgn * (a2 / 6 + b2 * 2.0 / 3.0 + c2 / 6.0) * theta
                s3 += sgn * (a3 / 6 + b3 * 2.0 / 3.0 + c3 / 6.0) * theta
            x1, y1, r1 = x2, y2, r2
            a0, a1, a2, a3 = c0, c1, c2, c3
    return s0, s1, s2, s3


@njit(cache=True)
def xint_fast(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, h, ndiv=NDIV, eps=E10,
              eps_r=E20):
    """xint with the fused polygon integrals (polyint_etas4). Same outputs up to rounding."""
    si, sid, siq, sigamma = polyint_etas4(b[7] ** 2, b[5] ** 2, b[6] ** 2, zmg[i], tx, ty, npoly,
                                          xx[i], yy[i], ndiv, eps, eps_r)
    return _xint_terms(i, b, zz, zmg, tz, tstart, h, si, sid, siq, sigamma)


@njit(cache=True)
def _xint_terms(i, b, zz, zmg, tz, tstart, h, si, sid, siq, sigamma):
    """Event i's offspring integral (returned) and gradient (into h), given its spatial
    integral si and that integral's D, q, gamma derivatives. Adds the Omori time part."""
    a2 = b[1] ** 2
    c = b[2] ** 2
    alfa = b[3] ** 2
    p = b[4] ** 2

    if zz[i] > tstart:
        ttemp = tz - zz[i]
        gi = 1.0 - (1.0 + ttemp / c) ** (1 - p)
        gic = -(1.0 - gi) * (1.0 - p) * (1.0 / (c + ttemp) - 1.0 / c)
        gip = -(1.0 - gi) * (math.log(c) - math.log(c + ttemp))
    else:
        ttemp2 = tz - zz[i]
        ttemp1 = tstart - zz[i]
        gi2 = 1 - (1.0 + ttemp2 / c) ** (1 - p)
        gic2 = -(1.0 - gi2) * (1.0 - p) * (1.0 / (c + ttemp2) - 1.0 / c)
        gip2 = -(1.0 - gi2) * (math.log(c) - math.log(c + ttemp2))
        gi1 = 1 - (1.0 + ttemp1 / c) ** (1 - p)
        gic1 = -(1.0 - gi1) * (1.0 - p) * (1.0 / (c + ttemp1) - 1.0 / c)
        gip1 = -(1.0 - gi1) * (math.log(c) - math.log(c + ttemp1))
        gi = gi2 - gi1
        gic = gic2 - gic1
        gip = gip2 - gip1

    sk = a2 * math.exp(alfa * zmg[i])
    fv = sk * gi * si
    h[0] = 0.0
    h[1] = sk * gi * si / a2 * 2.0 * b[1]
    h[2] = sk * gic * si * 2 * b[2]
    h[3] = sk * gi * si * zmg[i] * 2 * b[3]
    h[4] = sk * gip * si * 2 * b[4]
    h[5] = sk * gi * sid * 2 * b[5]
    h[6] = sk * gi * siq * 2 * b[6]
    h[7] = sk * gi * sigamma * 2 * b[7]
    return fv


@njit(cache=True)
def probcalc(x, zbkgd, xx, yy, zz, zmg, zprob):
    """zprob_i = mu*u_i / lambda_i: probability each event is a background event.
    (xlamb also computes the unused gradient, as in the original.)"""
    g = np.empty(8)
    for i in range(xx.shape[0]):
        bk = zbkgd[i] * x[0] * x[0]
        b2 = xlamb(i, x, zbkgd, xx, yy, zz, zmg, g)
        zprob[i] = bk / b2


@njit(cache=True)
def func15(b, ind, zbkgd, xint0, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, grad,
           fast_xint=False, ndiv=NDIV, eps=E10, eps_r=E20, log_floor=1e-25,
           log_penalty=100.0):
    """-log likelihood (returned) and its gradient w.r.t. sqrt-params b (into grad).
    Background u and xint0 are held fixed (from the last bkgdcalc).
    fast_xint: use the fused polygon integrals (xint_fast) instead of the 1:1 port.
    A target with lambda <= log_floor adds -log_penalty instead of log(lambda)."""
    nn = xx.shape[0]
    g = np.empty(8)
    ff = 0.0
    df = np.zeros(8)
    for i in range(nn):
        if ind[i] == 1:
            temp = xlamb(i, b, zbkgd, xx, yy, zz, zmg, g)
            if temp > log_floor:
                ff += math.log(temp)
            else:
                ff -= log_penalty
            for k in range(8):
                df[k] += g[k] / temp

    fv0 = 0.0
    hsum = np.zeros(8)
    for i in range(nn):
        if fast_xint:
            fv0 += xint_fast(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, g, ndiv, eps,
                             eps_r)
        else:
            fv0 += xint(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, g, ndiv, eps, eps_r)
        for k in range(8):
            hsum[k] += g[k]

    fv0 = fv0 + xint0 * b[0] * b[0]
    hsum[0] = xint0 * b[0] * 2
    for k in range(8):
        grad[k] = -df[k] + hsum[k]
    return -ff + fv0


@njit(cache=True)
def _rates_at(x0, y0, x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart):
    """(background, total-kernel density, full intensity) at grid point (x0, y0), time tz."""
    xmu = x[0] ** 2
    a2 = x[1] ** 2
    c = x[2] ** 2
    alfa = x[3] ** 2
    p = x[4] ** 2
    d = x[5] ** 2
    q = x[6] ** 2
    gamma = x[7] ** 2
    nn = xx.shape[0]
    s = 0.0
    s1 = 0.0
    for ii in range(nn):
        r0 = math.sqrt((xx[ii] - x0) ** 2 + (yy[ii] - y0) ** 2)
        s += zprob[ii] * dgauss(r0, zbandw[ii])
        s1 += dgauss(r0, zbandw[ii])
    lam = 0.0 + xmu * s / (tz - tstart)
    for ii in range(nn):
        delt = tz - zz[ii]
        pr1 = math.exp(alfa * zmg[ii])
        ssig = d * math.exp(gamma * zmg[ii])
        bbb = (q - 1) / PI / ssig
        dist2 = (xx[ii] - x0) ** 2 + (yy[ii] - y0) ** 2
        pr2 = (p - 1) / c * (1.0 + delt / c) ** (-p)
        pr3 = bbb * (dist2 / ssig + 1.0) ** (-q)
        lam += a2 * pr1 * pr2 * pr3
    return s / (tz - tstart), s1 / (tz - tstart), lam


@njit(cache=True)
def outrates_grid(x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart, gx, gy):
    """Background (prob-weighted), total-kernel density and full intensity at time tz on
    the grid gx x gy. Rows ordered x-major, like the original file."""
    ny = gy.shape[0]
    out = np.empty((gx.shape[0] * ny, 3))
    for ix in range(gx.shape[0]):
        for iy in range(ny):
            out[ix * ny + iy] = _rates_at(gx[ix], gy[iy], x, xx, yy, zz, zmg, zprob, zbandw,
                                          tz, tstart)
    return out


# ---------------------------------------------------------------------------------------
# Multithreaded versions (numba prange). Each event's contribution goes into its own array
# slot; the slots are then summed serially in index order, i.e. in exactly the order the
# serial loops above add them. Results are therefore bit-identical to the serial versions
# for any thread count. Extra memory: ~18 doubles per event.
# ---------------------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def bkgdcalc_par(xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, ndiv=NDIV,
                 eps=E10, eps_r=E20):
    """Parallel bkgdcalc (same result)."""
    nn = xx.shape[0]
    part = np.empty(nn)
    for i in prange(nn):
        s = 0.0
        for j in range(nn):
            r0 = math.sqrt((xx[j] - xx[i]) ** 2 + (yy[j] - yy[i]) ** 2)
            s += zprob[j] * dgauss(r0, zbandw[j])
        zbkgd[i] = s / (tz - tstart)
        w = np.empty(1)
        w[0] = zbandw[i]
        part[i] = zprob[i] * polyint(PGAUSS, w, tx, ty, npoly, xx[i], yy[i], ndiv, eps,
                                     eps_r)
    s = 0.0
    for i in range(nn):
        s += part[i]
    return s


@njit(cache=True, parallel=True)
def probcalc_par(x, zbkgd, xx, yy, zz, zmg, zprob):
    """Parallel probcalc (same result; events are independent)."""
    for i in prange(xx.shape[0]):
        g = np.empty(8)
        zprob[i] = zbkgd[i] * x[0] * x[0] / xlamb(i, x, zbkgd, xx, yy, zz, zmg, g)


@njit(cache=True, parallel=True)
def func15_par(b, ind, zbkgd, xint0, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, grad,
               fast_xint=False, ndiv=NDIV, eps=E10, eps_r=E20, log_floor=1e-25,
               log_penalty=100.0):
    """Parallel func15 (same result)."""
    nn = xx.shape[0]
    lam = np.ones(nn)
    lg = np.zeros((nn, 8))
    xi = np.empty(nn)
    xg = np.empty((nn, 8))
    for i in prange(nn):
        g = np.empty(8)
        if ind[i] == 1:
            lam[i] = xlamb(i, b, zbkgd, xx, yy, zz, zmg, g)
            for k in range(8):
                lg[i, k] = g[k]
        if fast_xint:
            xi[i] = xint_fast(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, g, ndiv, eps,
                              eps_r)
        else:
            xi[i] = xint(i, b, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, g, ndiv, eps,
                         eps_r)
        for k in range(8):
            xg[i, k] = g[k]
    return _func15_total(b, ind, xint0, grad, log_floor, log_penalty, lam, lg, xi, xg)


@njit(cache=True)
def _func15_total(b, ind, xint0, grad, log_floor, log_penalty, lam, lg, xi, xg):
    """-log L and gradient from per-event slots, summed in index order so that the result
    does not depend on the thread count."""
    nn = ind.shape[0]
    ff = 0.0
    df = np.zeros(8)
    fv0 = 0.0
    hsum = np.zeros(8)
    for i in range(nn):
        if ind[i] == 1:
            temp = lam[i]
            if temp > log_floor:
                ff += math.log(temp)
            else:
                ff -= log_penalty
            for k in range(8):
                df[k] += lg[i, k] / temp
    for i in range(nn):
        fv0 += xi[i]
        for k in range(8):
            hsum[k] += xg[i, k]
    fv0 = fv0 + xint0 * b[0] * b[0]
    hsum[0] = xint0 * b[0] * 2
    for k in range(8):
        grad[k] = -df[k] + hsum[k]
    return -ff + fv0


@njit(cache=True, parallel=True)
def outrates_grid_par(x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart, gx, gy):
    """Parallel outrates_grid (same result: grid points are independent)."""
    ny = gy.shape[0]
    out = np.empty((gx.shape[0] * ny, 3))
    for ix in prange(gx.shape[0]):
        for iy in range(ny):
            out[ix * ny + iy] = _rates_at(gx[ix], gy[iy], x, xx, yy, zz, zmg, zprob, zbandw,
                                          tz, tstart)
    return out


@njit(cache=True)
def outrates_points(x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart, px, py):
    """outrates_grid at arbitrary points (px[k], py[k]) (e.g. a lon/lat grid projected to
    an irregular set of model points)."""
    out = np.empty((px.shape[0], 3))
    for k in range(px.shape[0]):
        out[k] = _rates_at(px[k], py[k], x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart)
    return out


@njit(cache=True, parallel=True)
def outrates_points_par(x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart, px, py):
    """Parallel outrates_points (same result)."""
    out = np.empty((px.shape[0], 3))
    for k in prange(px.shape[0]):
        out[k] = _rates_at(px[k], py[k], x, xx, yy, zz, zmg, zprob, zbandw, tz, tstart)
    return out


@njit(cache=True)
def trigger_terms(i, x, zbkgd, xx, yy, zz, zmg, out):
    """The terms of lambda at event i (as outpmat in etas8p.f): out[j] = rate triggered by
    each earlier event j, returns the background term mu*u_i. O(i)."""
    a2 = x[1] ** 2
    c = x[2] ** 2
    alfa = x[3] ** 2
    p = x[4] ** 2
    d = x[5] ** 2
    q = x[6] ** 2
    gamma = x[7] ** 2
    for j in range(i):
        delt = zz[i] - zz[j]
        pr1 = math.exp(alfa * zmg[j])
        ssig = d * math.exp(gamma * zmg[j])
        bbb = (q - 1) / PI / ssig
        dist2 = (xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2
        pr2 = (p - 1) / c * (1.0 + delt / c) ** (-p)
        pr3 = bbb * (dist2 / ssig + 1.0) ** (-q)
        out[j] = a2 * pr1 * pr2 * pr3
    return zbkgd[i] * x[0] * x[0]


@njit(cache=True)
def _draw_parent(i, x, zbkgd, xx, yy, zz, zmg, u):
    """Inverse-CDF draw from (phi_i, rho_i0, ..., rho_i,i-1) with u in [0, 1): the parent's
    index, or -1 = background. Scaling u by lambda saves dividing every term by it."""
    rates = np.empty(i)
    s = trigger_terms(i, x, zbkgd, xx, yy, zz, zmg, rates)
    target = s
    for j in range(i):
        target += rates[j]
    target *= u
    if target < s:
        return -1
    for j in range(i):
        s += rates[j]
        if target < s:
            return j
    return i - 1                # only if rounding left target at the very top


@njit(cache=True)
def sample_parents(x, zbkgd, xx, yy, zz, zmg, u):
    """One parent per event from the uniforms u (Algorithm 3 of Zhuang et al. 2002)."""
    out = np.empty(xx.shape[0], dtype=np.int64)
    for i in range(xx.shape[0]):
        out[i] = _draw_parent(i, x, zbkgd, xx, yy, zz, zmg, u[i])
    return out


@njit(cache=True, parallel=True)
def sample_parents_par(x, zbkgd, xx, yy, zz, zmg, u):
    """Parallel sample_parents (same result: the uniforms are fixed before the loop)."""
    out = np.empty(xx.shape[0], dtype=np.int64)
    for i in prange(xx.shape[0]):
        out[i] = _draw_parent(i, x, zbkgd, xx, yy, zz, zmg, u[i])
    return out


# ---------------------------------------------------------------------------------------
# Polygon integrals by Gauss-Legendre quadrature in s ("s-GL", the default). Each edge adds
# sign * int F(r) dphi, with phi the angle seen from the kernel centre. Measuring position
# along the edge's line from the foot of the perpendicular, t = h sinh(s) gives
#     r = h cosh(s),    dphi = ds / cosh(s).
# F changes on a log scale of r, so it is smooth in s however close the centre is to the
# edge, and a few dozen nodes per edge replace thousands of equal-length pieces. No angle is
# ever computed from side lengths, which removes the rounding failure of acos at small
# angles. Nodes depend only on geometry, so -log L stays smooth in the parameters.
# Not bit-identical to the Fortran: the "original" preset keeps the equal-length pieces.
# ---------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def gauss_nodes(n):
    """Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


@njit(cache=True)
def _edge_sgl(x1, y1, x2, y2, xc, yc):
    """(sign, h, s at start, s at end). Sign 0: the centre is on the edge's line, no angle
    is swept and the edge adds nothing."""
    dx, dy = x2 - x1, y2 - y1
    L = math.sqrt(dx * dx + dy * dy)
    ax, ay, bx, by = x1 - xc, y1 - yc, x2 - xc, y2 - yc
    cross = ax * dy - ay * dx
    if cross == 0.0 or L == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    h = abs(cross) / L
    return ((-1.0 if cross < 0.0 else 1.0), h, math.asinh((ax * dx + ay * dy) / L / h),
            math.asinh((bx * dx + by * dy) / L / h))


@njit(cache=True)
def polyint_sgl4(gamma, d, q, xmag, xp, yp, npoly, xc, yc, gx, gw):
    """ETAS kernel over the polygon: the integral and its D, q, gamma derivatives."""
    ssig = d * math.exp(gamma * xmag)
    s0 = s1 = s2 = s3 = 0.0
    for j in range(npoly):
        sg, h, sa, sb = _edge_sgl(xp[j], yp[j], xp[j + 1], yp[j + 1], xc, yc)
        if sg == 0.0:
            continue
        ds = sb - sa
        mid = sa + ds * 0.5
        for k in range(gx.shape[0]):
            ch = math.cosh(mid + 0.5 * ds * gx[k])
            a0, a1, a2, a3 = _fr4(h * ch, ssig, d, q, xmag)
            wt = sg * 0.5 * ds * gw[k] / ch
            s0 += wt * a0
            s1 += wt * a1
            s2 += wt * a2
            s3 += wt * a3
    return s0, s1, s2, s3


@njit(cache=True)
def polyint_gauss_sgl(w, xp, yp, npoly, xc, yc, gx, gw):
    """Gaussian background kernel of bandwidth w over the polygon. F is close to a step in
    s here, since r^2 grows like cosh^2(s), but real bandwidths need about as many nodes
    as the ETAS kernel; see configs/jma.toml."""
    s = 0.0
    for j in range(npoly):
        sg, h, sa, sb = _edge_sgl(xp[j], yp[j], xp[j + 1], yp[j + 1], xc, yc)
        if sg != 0.0:
            ds = sb - sa
            mid = sa + 0.5 * ds
            for k in range(gx.shape[0]):
                ch = math.cosh(mid + 0.5 * ds * gx[k])
                s += (sg * 0.5 * ds * gw[k] / ch
                      * ((1.0 - math.exp(-(h * h * ch * ch) / 2.0 / (w * w))) / 2.0 / PI))
    return s


@njit(cache=True)
def _bkgd_sgl_event(i, xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, gx, gw):
    """Sets zbkgd[i]; returns event i's term of xint0."""
    s = 0.0
    for j in range(xx.shape[0]):
        r0 = math.sqrt((xx[j] - xx[i]) ** 2 + (yy[j] - yy[i]) ** 2)
        s += zprob[j] * dgauss(r0, zbandw[j])
    zbkgd[i] = s / (tz - tstart)
    return zprob[i] * polyint_gauss_sgl(zbandw[i], tx, ty, npoly, xx[i], yy[i], gx, gw)


@njit(cache=True)
def bkgdcalc_sgl(xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, gx, gw):
    """bkgdcalc with s-GL polygon integrals."""
    s = 0.0
    for i in range(xx.shape[0]):
        s += _bkgd_sgl_event(i, xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, gx, gw)
    return s


@njit(cache=True, parallel=True)
def bkgdcalc_sgl_par(xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd, gx, gw):
    """Parallel bkgdcalc_sgl (same result)."""
    nn = xx.shape[0]
    part = np.empty(nn)
    for i in prange(nn):
        part[i] = _bkgd_sgl_event(i, xx, yy, zprob, zbandw, tx, ty, npoly, tz, tstart, zbkgd,
                                  gx, gw)
    s = 0.0
    for i in range(nn):
        s += part[i]
    return s


# ---------------------------------------------------------------------------------------
# Cheaper lambda sums, with every pair kept. Cutoffs do not work here: Omori decay is too
# slow (p - 1 = 0.15 on JMA) for dropped pairs to stay negligible. Used with s-GL only; the
# 1:1 xlamb above stays for the bit-exact "original" path.
# ---------------------------------------------------------------------------------------

@njit(cache=True)
def _parent_terms(b, zmg):
    """exp(alpha m_j), ssig_j and (q-1)/ssig_j/pi depend only on the parent, so they are
    computed once per call rather than once per pair. Same expressions as xlamb."""
    alfa, d, q, gamma = b[3] ** 2, b[5] ** 2, b[6] ** 2, b[7] ** 2
    n = zmg.shape[0]
    pr1, ssig, bbb = np.empty(n), np.empty(n), np.empty(n)
    for j in range(n):
        pr1[j] = math.exp(alfa * zmg[j])
        ssig[j] = d * math.exp(gamma * zmg[j])
        bbb[j] = (q - 1) / ssig[j] / PI
    return pr1, ssig, bbb


@njit(cache=True)
def xlamb_cache(i, b, zbkgd, xx, yy, zz, zmg, g1, pr1c, ssigc, bbbc):
    """xlamb with _parent_terms: bit-identical to xlamb."""
    xmu = b[0] ** 2
    a2 = b[1] ** 2
    c = b[2] ** 2
    p = b[4] ** 2
    d = b[5] ** 2
    q = b[6] ** 2
    s = xmu * zbkgd[i]
    sg1 = zbkgd[i]
    sg2 = sg3 = sg4 = sg5 = sg6 = sg7 = sg8 = 0.0
    for j in range(i):
        delt = zz[i] - zz[j]
        pr1 = pr1c[j]
        pr2 = (p - 1) / c * (1.0 + delt / c) ** (-p)
        ssig = ssigc[j]
        dist2 = (xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2
        pr3 = bbbc[j] * (dist2 / ssig + 1.0) ** (-q)

        s += a2 * pr1 * pr2 * pr3
        sg2 += pr1 * pr2 * pr3
        pr2_c = pr2 * (-1.0 / c - p / (c + delt) + p / c)
        sg3 += a2 * pr1 * pr2_c * pr3
        pr1_alfa = pr1 * zmg[j]
        sg4 += a2 * pr1_alfa * pr2 * pr3
        pr2_p = pr2 * (1.0 / (p - 1) - math.log(1.0 + delt / c))
        sg5 += a2 * pr1 * pr2_p * pr3
        pr3_d = pr3 / d * (-1.0 + q * (1.0 - 1.0 / (1.0 + dist2 / ssig)))
        sg6 += a2 * pr1 * pr2 * pr3_d
        pr3_q = pr3 * (1.0 / (q - 1) - math.log(1 + dist2 / ssig))
        sg7 += a2 * pr1 * pr2 * pr3_q
        pr3_gamma = pr3 * (-zmg[j] + q * zmg[j] * (1.0 - 1.0 / (1.0 + dist2 / ssig)))
        sg8 += a2 * pr1 * pr2 * pr3_gamma
    g1[0] = sg1 * 2 * b[0]
    g1[1] = sg2 * 2 * b[1]
    g1[2] = sg3 * 2 * b[2]
    g1[3] = sg4 * 2 * b[3]
    g1[4] = sg5 * 2 * b[4]
    g1[5] = sg6 * 2 * b[5]
    g1[6] = sg7 * 2 * b[6]
    g1[7] = sg8 * 2 * b[7]
    return s


@njit(cache=True)
def xlamb_logs(i, b, zbkgd, xx, yy, zz, zmg, g1, pr1c, ssigc, bbbc):
    """xlamb_cache that also reuses the logs the gradient needs: x^-p y^-q = exp(-p log x -
    q log y), so 2 pow + 2 log become 2 log + 1 exp. About twice as fast. Rounding differs:
    lambda by at most ~10 ulp on JMA."""
    xmu = b[0] ** 2
    a2 = b[1] ** 2
    c = b[2] ** 2
    p = b[4] ** 2
    d = b[5] ** 2
    q = b[6] ** 2
    pc = (p - 1) / c
    t1, t2, t3 = -1.0 / c + p / c, 1.0 / (p - 1), 1.0 / (q - 1)
    s = xmu * zbkgd[i]
    sg1 = zbkgd[i]
    sg2 = sg3 = sg4 = sg5 = sg6 = sg7 = sg8 = 0.0
    for j in range(i):
        delt = zz[i] - zz[j]
        l1 = math.log(1.0 + delt / c)
        mj = zmg[j]
        ratio = (((xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2) / ssigc[j])
        l2 = math.log(1.0 + ratio)
        tt = pr1c[j] * pc * bbbc[j] * math.exp(-p * l1 - q * l2)
        t = a2 * tt
        v = q * (1.0 - 1.0 / (1.0 + ratio))
        s += t
        sg2 += tt
        sg3 += t * (t1 - p / (c + delt))
        sg4 += t * mj
        sg5 += t * (t2 - l1)
        sg6 += t / d * (-1.0 + v)
        sg7 += t * (t3 - l2)
        sg8 += t * mj * (v - 1.0)
    g1[0] = sg1 * 2 * b[0]
    g1[1] = sg2 * 2 * b[1]
    g1[2] = sg3 * 2 * b[2]
    g1[3] = sg4 * 2 * b[3]
    g1[4] = sg5 * 2 * b[4]
    g1[5] = sg6 * 2 * b[5]
    g1[6] = sg7 * 2 * b[6]
    g1[7] = sg8 * 2 * b[7]
    return s


@njit(cache=True)
def _xlamb_value(i, b, zbkgd, xx, yy, zz, pr1c, ssigc, bbbc):
    """xlamb's value without the gradient: bit-identical value."""
    a2, c, p, q = b[1] ** 2, b[2] ** 2, b[4] ** 2, b[6] ** 2
    s = b[0] ** 2 * zbkgd[i]
    for j in range(i):
        delt = zz[i] - zz[j]
        pr2 = (p - 1) / c * (1.0 + delt / c) ** (-p)
        dist2 = (xx[i] - xx[j]) ** 2 + (yy[i] - yy[j]) ** 2
        pr3 = bbbc[j] * (dist2 / ssigc[j] + 1.0) ** (-q)
        s += a2 * pr1c[j] * pr2 * pr3
    return s


@njit(cache=True)
def probcalc_value(x, zbkgd, xx, yy, zz, zmg, zprob):
    """probcalc without the unused gradient (same result)."""
    pr1c, ssigc, bbbc = _parent_terms(x, zmg)
    for i in range(xx.shape[0]):
        zprob[i] = zbkgd[i] * x[0] * x[0] / _xlamb_value(i, x, zbkgd, xx, yy, zz, pr1c, ssigc, bbbc)


@njit(cache=True, parallel=True)
def probcalc_value_par(x, zbkgd, xx, yy, zz, zmg, zprob):
    """Parallel probcalc_value (same result)."""
    pr1c, ssigc, bbbc = _parent_terms(x, zmg)
    for i in prange(xx.shape[0]):
        zprob[i] = zbkgd[i] * x[0] * x[0] / _xlamb_value(i, x, zbkgd, xx, yy, zz, pr1c, ssigc, bbbc)


@njit(cache=True)
def _func15_sgl_event(i, b, ind, zbkgd, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, gx, gw, lam,
                      lg, xi, xg, use_logs, pr1c, ssigc, bbbc):
    """Fills event i's slots for _func15_total."""
    g = np.empty(8)
    if ind[i] == 1:
        if use_logs:
            lam[i] = xlamb_logs(i, b, zbkgd, xx, yy, zz, zmg, g, pr1c, ssigc, bbbc)
        else:
            lam[i] = xlamb_cache(i, b, zbkgd, xx, yy, zz, zmg, g, pr1c, ssigc, bbbc)
        for k in range(8):
            lg[i, k] = g[k]
    si, sid, siq, sigamma = polyint_sgl4(b[7] ** 2, b[5] ** 2, b[6] ** 2, zmg[i], tx, ty, npoly,
                                         xx[i], yy[i], gx, gw)
    xi[i] = _xint_terms(i, b, zz, zmg, tz, tstart, g, si, sid, siq, sigamma)
    for k in range(8):
        xg[i, k] = g[k]


@njit(cache=True)
def func15_sgl(b, ind, zbkgd, xint0, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, grad, gx, gw,
               log_floor=1e-25, log_penalty=100.0, use_logs=True):
    """func15 with s-GL polygon integrals. use_logs: xlamb_logs, else xlamb_cache."""
    nn = xx.shape[0]
    lam, lg, xi, xg = np.ones(nn), np.zeros((nn, 8)), np.empty(nn), np.empty((nn, 8))
    pr1c, ssigc, bbbc = _parent_terms(b, zmg)
    for i in range(nn):
        _func15_sgl_event(i, b, ind, zbkgd, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, gx, gw,
                          lam, lg, xi, xg, use_logs, pr1c, ssigc, bbbc)
    return _func15_total(b, ind, xint0, grad, log_floor, log_penalty, lam, lg, xi, xg)


@njit(cache=True, parallel=True)
def func15_sgl_par(b, ind, zbkgd, xint0, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, grad, gx, gw,
                   log_floor=1e-25, log_penalty=100.0, use_logs=True):
    """Parallel func15_sgl (same result)."""
    nn = xx.shape[0]
    lam, lg, xi, xg = np.ones(nn), np.zeros((nn, 8)), np.empty(nn), np.empty((nn, 8))
    pr1c, ssigc, bbbc = _parent_terms(b, zmg)
    for i in prange(nn):
        _func15_sgl_event(i, b, ind, zbkgd, xx, yy, zz, zmg, tx, ty, npoly, tz, tstart, gx, gw,
                          lam, lg, xi, xg, use_logs, pr1c, ssigc, bbbc)
    return _func15_total(b, ind, xint0, grad, log_floor, log_penalty, lam, lg, xi, xg)
