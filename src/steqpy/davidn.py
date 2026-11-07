"""Port of davidn.f: Davidon-Fletcher-Powell quasi-Newton minimiser with the ISM line search.

The Fortran is goto-driven; this keeps its control flow and arithmetic order (plain loops,
no numpy reductions) so iterates match. `funct(x) -> (f, grad)`, and optionally
`value(x) -> f`, which must return the same bits as funct's f.

Unreachable branches dropped: the original `funct` (func15) never sets its error flag, so
`ig` only becomes 1 inside `linear` (degenerate parabola). The "label 500" retry path needs
`ig == 1` right after an expansion step, which cannot happen, so it is omitted.

Added to the Fortran: a NaN or infinite f counts as "went up", and a NaN or infinite f or gradient
at an accepted point ends the search. The Fortran can loop forever on NaN (`e3 > e2` is false).
Neither changes anything while every value is finite.
"""
import math


def _step(x, h, ram):
    return [x[i] + ram * h[i] for i in range(len(x))]


def linear(x, h, ram, ee, ig, value, log, shrink=0.1, expand=2.0):
    """Line search along h from x. Returns (ram, ee, ig): step, f at the last trial, flag.
    Brackets the minimum by shrinking (x shrink) or expanding (x expand) the step, then
    takes the vertex of the parabola through the bracket (twice)."""
    def trial(xv):
        e = value(xv, log)
        return e if math.isfinite(e) else math.inf

    if ram <= 1.0e-30:
        ram = 0.1
    const2 = 1.0e-16
    hnorm = 0.0
    for v in h:
        hnorm += v ** 2
    hnorm = math.sqrt(hnorm)
    if not math.isfinite(hnorm):
        return 0.0, ee, ig
    if hnorm > 1:
        ram = ram / hnorm

    ram2 = ram
    e1 = ee
    ram1 = 0.0
    e2 = trial(_step(x, h, ram2))

    if ig == 1 or e2 > e1:
        # label 50: shrink until below e1
        while True:
            ram3 = ram2
            e3 = e2
            ram2 = ram3 * shrink
            if ram2 * hnorm < const2:
                return 0.0, ee, ig
            e2 = trial(_step(x, h, ram2))
            if not e2 > e1:
                break
    else:
        # label 30: expand until the value goes up
        while True:
            ram3 = ram2 * expand
            e3 = trial(_step(x, h, ram3))
            if e3 > e2:
                break
            ram1 = ram2
            ram2 = ram3
            e1 = e2
            e2 = e3

    # label 70 -> sub1 (parabola through the three bracketing points)
    ram, ok = _parabola(ram1, ram2, ram3, e1, e2, e3)
    if not ok:
        return ram2, ee, 1
    ee = trial(_step(x, h, ram))  # label 80

    # label 95: replace one bracket point with the new trial, fit again
    if ram > ram2:
        if ee <= e2:
            ram1, ram2, e1, e2 = ram2, ram, e2, ee
        else:
            ram3, e3 = ram, ee
    else:
        if ee >= e2:
            ram1, e1 = ram, ee
        else:
            ram3, ram2, e3, e2 = ram2, ram, e2, ee
    ram, ok = _parabola(ram1, ram2, ram3, e1, e2, e3)
    if not ok:
        return ram2, ee, 1
    ee = trial(_step(x, h, ram))  # label 130
    if e2 < ee:
        ram = ram2
    return ram, ee, ig


def _parabola(ram1, ram2, ram3, e1, e2, e3):
    """Fortran sub1: vertex of the parabola through (ram_k, e_k); ok=False if degenerate."""
    a1 = (ram3 - ram2) * e1
    a2 = (ram1 - ram3) * e2
    a3 = (ram2 - ram1) * e3
    b2 = (a1 + a2 + a3) * 2.0
    b1 = a1 * (ram3 + ram2) + a2 * (ram1 + ram3) + a3 * (ram2 + ram1)
    if b2 == 0.0 or not math.isfinite(b1 / b2):
        return ram2, False
    return b1 / b2, True


def _usable(f, g):
    return math.isfinite(f) and all(math.isfinite(v) for v in g)


def davidn(x, funct, log, tol=1.0e-6, max_sweeps=10, step=0.05, shrink=0.1, expand=2.0, value=None):
    """Minimise funct starting at x (list, modified in place). At most max_sweeps sweeps of
    n steps. tol: stopping tolerance on gradient, step and relative f change (original 1e-6).
    step: first line-search step; shrink/expand: line-search bracketing factors.
    value: cheaper f for the line search, which never uses the gradient.
    The last funct call is always at the returned x."""
    if value is None:
        value = lambda xv, lg: funct(xv, lg)[0]
    n = len(x)
    tau1 = tau2 = tol
    eps1 = eps2 = tol
    ramda = step
    const1 = 1.0e-17
    h = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    s = [0.0] * n
    dx = [0.0] * n
    g0 = [0.0] * n
    y = [0.0] * n
    wrk = [0.0] * n
    ig = 0  # uninitialised in the Fortran; assumed 0

    xm, g = funct(x, log)
    log(f" - log likelihood = {xm:23.15E}     aic = {2 * (xm + n):10.1f}")
    if not _usable(xm, g):
        log(" f or gradient is not finite at the start: nothing to do")
        return x

    icc = 0
    done = False
    while not done:
        icc += 1
        for ic in range(1, n + 1):
            if not (ic == 1 and icc == 1):
                for i in range(n):
                    y[i] = g[i] - g0[i]
                for i in range(n):
                    sm = 0.0
                    for j in range(n):
                        sm = sm + y[j] * h[i][j]
                    wrk[i] = sm
                s1 = 0.0
                s2 = 0.0
                for i in range(n):
                    s1 = s1 + wrk[i] * y[i]
                    s2 = s2 + dx[i] * y[i]
                if s1 <= const1 or s2 <= const1:
                    done = True
                    break
                if s1 > s2:
                    # Davidon-Fletcher-Powell correction
                    for i in range(n):
                        for j in range(i, n):
                            h[i][j] = h[i][j] + dx[i] * dx[j] / s2 - wrk[i] * wrk[j] / s1
                            h[j][i] = h[i][j]
                else:
                    # Fletcher correction
                    stem = s1 / s2 + 1.0
                    for i in range(n):
                        for j in range(i, n):
                            h[i][j] = h[i][j] - (dx[i] * wrk[j] + wrk[i] * dx[j]
                                                 - dx[i] * dx[j] * stem) / s2
                            h[j][i] = h[i][j]

            # label 120: search direction s = -H g
            ss = 0.0
            for i in range(n):
                sm = 0.0
                for j in range(n):
                    sm = sm + h[i][j] * g[j]
                ss = ss + sm * sm
                s[i] = -sm
            s1 = 0.0
            s2 = 0.0
            for i in range(n):
                s1 = s1 + s[i] * g[i]
                s2 = s2 + g[i] * g[i]
            ds2 = math.sqrt(s2)
            gtem = abs(s1) / ds2
            if gtem <= tau1 and ds2 <= tau2:
                done = True
                break
            if s1 >= 0.0:  # not a descent direction: reset H to identity
                for i in range(n):
                    for j in range(n):
                        h[i][j] = 0.0
                    h[i][i] = 1.0
                    s[i] = -s[i]

            ed = xm
            ramda, ed, ig = linear(x, s, ramda, ed, ig, value, log, shrink, expand)
            log(f"  lmbd = {ramda:13.7E}  -ll = {ed:21.15E}  {s1:9.2E} {s2:9.2E}")

            s1 = 0.0
            for i in range(n):
                dx[i] = s[i] * ramda
                s1 = s1 + dx[i] * dx[i]
                g0[i] = g[i]
                x[i] = x[i] + dx[i]
            xmb = xm
            xm, g = funct(x, log)
            if not _usable(xm, g):
                log(" f or gradient is not finite: back to the last good point, stopping")
                for i in range(n):
                    x[i] = x[i] - dx[i]
                xm, g = funct(x, log)
                done = True
                break

            s2 = 0.0
            for i in range(n):
                s2 = s2 + g[i] * g[i]
            if math.sqrt(s2) > tau2:
                continue
            if xmb / xm - 1.0 < eps1 and math.sqrt(s1) < eps2:
                done = True
                break
        if icc >= max_sweeps:
            done = True

    log(f" - log likelihood = {xm:23.15E}     aic = {2 * (xm + n):10.1f}")
    return x
