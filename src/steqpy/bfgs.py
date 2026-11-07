"""BFGS with a strong Wolfe line search (Nocedal & Wright, algorithms 3.5, 3.6 and 6.1), the
default optimiser in place of davidn (DFP with a value-only line search, the original's).

Two choices matter, both found by testing 9 starts x 5 variants on JMA
(docs/optimizer_hypotheses.py, docs/results_outer_objective.txt):
- The inverse Hessian H is kept between outer iterations (the background changes very little
  between them), so later outers start with good curvature: about 90 calls per fit against
  davidn's about 540.
- The first trial of every line search moves at most `step` in sqrt(params), as davidn's does.
  A full step with H = identity (at the start, or after a reset) is -g, and |g| is in the
  thousands: on one JMA start that step was 17000x the Newton step and sent p to 2.5e5.
"""
import math

import numpy as np

C1, C2 = 1e-4, 0.9      # sufficient decrease, curvature (usual quasi-Newton values)
MAX_EVALS = 25          # per line search
FTOL = 1e-13            # stop when the predicted decrease -g.p/2 < FTOL |f| twice running


def _cubic(a0, f0, d0, a1, f1, d1):
    """Minimiser of the cubic through both ends' value and slope; None if it has none."""
    d = d0 + d1 - 3.0 * (f0 - f1) / (a0 - a1)
    disc = d * d - d0 * d1
    if not math.isfinite(disc) or disc < 0.0:
        return None
    root = math.copysign(math.sqrt(disc), a1 - a0)
    den = d1 - d0 + 2.0 * root
    return None if den == 0.0 else a1 - (a1 - a0) * (d1 + root - d) / den


def wolfe(phi, f0, d0, a1):
    """Step a along a descent direction from f0 with slope d0 < 0; phi(a) = (f, slope, g).
    Returns (a, f, g) meeting the strong Wolfe conditions, the best point found, or None.
    A NaN f fails the decrease test, so the step shrinks back."""
    evals = 0

    def zoom(lo, f_lo, d_lo, hi, f_hi, d_hi):
        nonlocal evals
        best = None
        while evals < MAX_EVALS:
            a = None
            if math.isfinite(f_hi) and math.isfinite(d_hi):
                a = _cubic(lo, f_lo, d_lo, hi, f_hi, d_hi)
            left, right = min(lo, hi), max(lo, hi)
            margin = 0.1 * (right - left)
            if a is None or not (left + margin <= a <= right - margin):
                a = 0.5 * (lo + hi)
            f, d, g = phi(a)
            evals += 1
            if not f <= f0 + C1 * a * d0 or f >= f_lo:
                hi, f_hi, d_hi = a, f, d
            else:
                best = (a, f, g)
                if abs(d) <= -C2 * d0:
                    return best
                if d * (hi - lo) >= 0.0:
                    hi, f_hi, d_hi = lo, f_lo, d_lo
                lo, f_lo, d_lo = a, f, d
        return best

    a_prev, f_prev, d_prev = 0.0, f0, d0
    a = a1
    first = True
    while evals < MAX_EVALS:
        f, d, g = phi(a)
        evals += 1
        if not f <= f0 + C1 * a * d0 or (not first and f >= f_prev):
            return zoom(a_prev, f_prev, d_prev, a, f, d)
        if abs(d) <= -C2 * d0:
            return a, f, g
        if d >= 0.0:
            return zoom(a, f, d, a_prev, f_prev, d_prev)
        a_prev, f_prev, d_prev = a, f, d
        a = 2.0 * a
        first = False
    return None


def bfgs(x, funct, log, H=None, tol=1e-6, max_sweeps=10, step=0.05):
    """Minimise funct(x, log) -> (f, g) from x (list, updated in place). H: inverse Hessian
    from the previous outer iteration, or None. Returns the H to pass to the next call."""
    n = len(x)
    xk = np.array(x, dtype=float)
    fk, g = funct(list(xk), log)
    g = np.array(g)
    last = xk.copy()
    have_h = H is not None
    H = np.array(H, dtype=float) if have_h else np.eye(n)
    passes = 0
    for _ in range(max_sweeps * n):
        p = -H @ g
        d0 = float(g @ p)
        if not d0 < 0.0:                    # not a descent direction (or NaN): restart
            H, p, d0, have_h = np.eye(n), -g, -float(g @ g), False
        if not np.linalg.norm(g) > tol:
            break
        passes = passes + 1 if -0.5 * d0 <= FTOL * abs(fk) else 0
        if passes >= 2:
            break
        a1 = min(1.0, step / float(np.linalg.norm(p)))

        def phi(a):
            nonlocal last
            last = xk + a * p
            f, gr = funct(list(last), log)
            gr = np.array(gr)
            return f, float(gr @ p), gr

        r = wolfe(phi, fk, d0, a1)
        if r is None:
            if have_h:                      # stale curvature: retry along -g
                H, have_h = np.eye(n), False
                continue
            break
        a, f_new, g_new = r
        s, y = a * p, g_new - g
        xk, fk, g = xk + s, f_new, g_new
        ys = float(y @ s)
        if ys > 1e-14 * float(np.linalg.norm(y) * np.linalg.norm(s)):
            if not have_h:                  # first update: scale identity to the curvature seen
                H = np.eye(n) * ys / float(y @ y)
                have_h = True
            rho = 1.0 / ys
            A = np.eye(n) - rho * np.outer(s, y)
            H = A @ H @ A.T + rho * np.outer(s, s)
    if not np.array_equal(last, xk):
        funct(list(xk), log)                # leave the last call at the returned x
    log(f" bfgs: -log L {fk:.10f}, |g| {float(np.linalg.norm(g)):.3e}")
    x[:] = list(xk)
    return H
