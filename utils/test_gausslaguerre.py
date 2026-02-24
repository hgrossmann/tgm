#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable accuracy test for d=2 jump-kernel moments (mu^J and tr(Cov)).

- "Reference" integrator: adaptive Simpson on [0, R_max] in float64 (CPU).
- "Production" integrator: fixed-N Gauss–Laguerre (N=64) with r = sqrt(2x) mapping.
- Tests 10 ||a_t|| values on a log scale (plus zero), random directions.
- Prints absolute & relative errors.

Usage:
  python utils/test_jump_moments_stability.py --tau 0.3 --N 64 --tol 1e-10
"""

import argparse
import math
import numpy as np

# ------------------------------
# Problem-specific definitions
# ------------------------------

D = 2  # dimension fixed

def r_bounds_for_a(a_norm):
    """
    r_-(a), r_+(a) from the proposition for d=2.
    """
    s = math.sqrt(a_norm*a_norm + D)
    return s - a_norm, s + a_norm

def phi_of(r, a_norm):
    """
    phi(r,a) = arccos( (d - r^2) / (2 ||a|| r) ) where defined.
    Caller must ensure 2||a||r > 0 and r within (r_-, r_+).
    """
    denom = 2.0 * a_norm * r
    if denom <= 0.0:
        return 0.0  # not used outside mid-region
    arg = (D - r*r) / denom
    # clamp for numerical safety
    if arg < -1.0: arg = -1.0
    if arg >  1.0: arg =  1.0
    return math.acos(arg)

def f0_f1_f2(r, a_norm):
    """
    Evaluate f0, f1, f2 at scalar r for a given ||a||, d=2.
    Piecewise per the proposition.
    """
    r_minus, r_plus = r_bounds_for_a(a_norm)

    if r <= r_minus:
        return 0.0, 0.0, 0.0

    if r >= r_plus:
        # hi region
        f0 = 2.0*math.pi * (r*r - D)
        f1 = 2.0*math.pi * a_norm * r
        f2 = math.pi * (r*r - D)
        return f0, f1, f2

    # mid region
    # denominator in phi is >0 here because r in (r_-, r_+), a_norm>=0
    phi = phi_of(r, a_norm)
    sin_phi  = math.sin(phi)
    sin2_phi = math.sin(2.0*phi)
    sin3_phi = math.sin(3.0*phi)

    f0 = 2.0*phi + 4.0*a_norm*r * sin_phi
    f1 = 2.0*(r*r - D)*sin_phi + 2.0*a_norm*r * (phi + 0.5*sin2_phi)
    f2 = (r*r - D)*(phi + 0.5*sin2_phi) + 2.0*a_norm*r * (1.5*sin_phi + (1.0/6.0)*sin3_phi)
    return f0, f1, f2

def integrands(r, a_norm):
    """
    Return the three integrands (scalar) for the required integrals:
      I0 = ∫ e^{-r^2/2} r   f0(r) dr
      I1 = ∫ e^{-r^2/2} r^2 f1(r) dr
      I2 = ∫ e^{-r^2/2} r^3 f2(r) dr
    """
    f0, f1, f2 = f0_f1_f2(r, a_norm)
    e = math.exp(-0.5 * r*r)
    r1 = r
    r2 = r*r
    r3 = r2*r
    return e * r1 * f0, e * r2 * f1, e * r3 * f2

# ------------------------------
# Robust 1D adaptive Simpson
# ------------------------------

def _simpson(fa, fm, fb, a, b):
    return (b - a) * (fa + 4.0*fm + fb) / 6.0

def adaptive_simpson(f, a, b, fa=None, fm=None, fb=None, S=None, tol=1e-10, depth=0, max_depth=30):
    """
    Adaptive Simpson integration of scalar function f over [a,b].
    Returns (value, eval_count).
    """
    if fa is None:
        fa = f(a)
    if fb is None:
        fb = f(b)
    if fm is None:
        m  = 0.5*(a+b)
        fm = f(m)
    else:
        m = 0.5*(a+b)
    if S is None:
        S = _simpson(fa, fm, fb, a, b)

    # Subdivide
    lm = 0.5*(a + m)
    rm = 0.5*(m + b)
    flm = f(lm)
    frm = f(rm)

    S_left  = _simpson(fa, flm, fm, a, m)
    S_right = _simpson(fm, frm, fb, m, b)
    S2 = S_left + S_right

    # Error estimate
    err = abs(S2 - S)

    # Tolerance scaled (Richardson extrapolation improves order)
    if (depth >= max_depth) or (err < 15.0 * tol):
        # Add correction term (S2 - S)/15 for Richardson extrapolation
        return S2 + (S2 - S)/15.0, 2  # two new evals (flm, frm)

    left_val,  left_evals  = adaptive_simpson(f, a, m,  fa, flm, fm, S_left,  tol*0.5, depth+1, max_depth)
    right_val, right_evals = adaptive_simpson(f, m, b,  fm, frm, fb, S_right, tol*0.5, depth+1, max_depth)
    return left_val + right_val, (2 + left_evals + right_evals)

def integrate_reference(a_norm, tol=1e-10):
    """
    Reference integrals using adaptive Simpson on [0, R_max], float64.
    Picks R_max from tail bound and support requirement.

      I0 = ∫ e^{-r^2/2} r   f0(r) dr
      I1 = ∫ e^{-r^2/2} r^2 f1(r) dr
      I2 = ∫ e^{-r^2/2} r^3 f2(r) dr
    """
    # Tail: choose R_tail such that exp(-R^2/2) < tol
    # exp(-R^2/2) = tol  => R = sqrt(2 log(1/tol))
    R_tail = math.sqrt(2.0 * max(0.0, math.log(1.0 / max(tol, 1e-300))))
    # ensure we cover support up to r_+(a)
    _, r_plus = r_bounds_for_a(a_norm)
    R_max = max(R_tail + 3.0, r_plus + 3.0)  # safety margin

    # define three scalar integrands
    f0 = lambda r: integrands(r, a_norm)[0]
    f1 = lambda r: integrands(r, a_norm)[1]
    f2 = lambda r: integrands(r, a_norm)[2]

    # initial function values at endpoints/mid
    a = 0.0
    b = R_max
    m = 0.5*(a+b)

    # integrate each with same subdivision to reuse evaluations a bit
    fa0, fm0, fb0 = f0(a), f0(m), f0(b)
    S0 = _simpson(fa0, fm0, fb0, a, b)
    I0, _ = adaptive_simpson(f0, a, b, fa0, fm0, fb0, S0, tol=tol)

    fa1, fm1, fb1 = f1(a), f1(m), f1(b)
    S1 = _simpson(fa1, fm1, fb1, a, b)
    I1, _ = adaptive_simpson(f1, a, b, fa1, fm1, fb1, S1, tol=tol)

    fa2, fm2, fb2 = f2(a), f2(m), f2(b)
    S2 = _simpson(fa2, fm2, fb2, a, b)
    I2, _ = adaptive_simpson(f2, a, b, fa2, fm2, fb2, S2, tol=tol)

    return I0, I1, I2

# ------------------------------------
# Fixed-N Gauss–Laguerre (small N)
# ------------------------------------

def laggauss_smallN(n):
    """
    Safe path for small n: use NumPy's laggauss (float64).
    """
    x, w = np.polynomial.laguerre.laggauss(int(n))
    # sanity: sum of weights should be ≈ 1 for ∫_0^∞ e^{-x} dx
    if not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-8:
        raise RuntimeError("laggauss returned unstable weights.")
    return x, w

def integrate_gl64(a_norm, N=64):
    """
    Production-like fixed quadrature with N Gauss–Laguerre nodes (alpha=0),
    using mapping x = r^2/2. float64 on CPU.
    """
    x, w = laggauss_smallN(N)  # x,w in float64
    r = np.sqrt(2.0 * x)

    # vectorized evaluation
    f0_vals = np.zeros_like(r)
    f1_vals = np.zeros_like(r)
    f2_vals = np.zeros_like(r)

    # hi/mid/low masks
    r_minus, r_plus = r_bounds_for_a(a_norm)
    hi  = r >= r_plus
    mid = (r > r_minus) & (r < r_plus)

    # hi region
    if np.any(hi):
        rr = r[hi]
        f0_vals[hi] = 2.0*np.pi * (rr*rr - D)
        f1_vals[hi] = 2.0*np.pi * a_norm * rr
        f2_vals[hi] = np.pi * (rr*rr - D)

    # mid region
    if np.any(mid):
        rr = r[mid]
        denom = 2.0 * max(a_norm, 0.0) * rr
        arg = (D - rr*rr) / np.maximum(denom, 1e-300)
        arg = np.clip(arg, -1.0, 1.0)
        phi = np.arccos(arg)
        sin_phi  = np.sin(phi)
        sin2_phi = np.sin(2.0*phi)
        sin3_phi = np.sin(3.0*phi)

        f0_vals[mid] = 2.0*phi + 4.0*a_norm*rr * sin_phi
        f1_vals[mid] = 2.0*(rr*rr - D)*sin_phi + 2.0*a_norm*rr * (phi + 0.5*sin2_phi)
        f2_vals[mid] = (rr*rr - D)*(phi + 0.5*sin2_phi) + 2.0*a_norm*rr * (1.5*sin_phi + (1.0/6.0)*sin3_phi)

    # Gauss–Laguerre sums with the mapping:
    # ∫ e^{-r^2/2} r f0 dr = ∑ w * f0
    # ∫ e^{-r^2/2} r^2 f1 dr = ∑ w * sqrt(2x) f1
    # ∫ e^{-r^2/2} r^3 f2 dr = ∑ w * (2x) f2
    I0 = np.sum(w * f0_vals)
    I1 = np.sum(w * (np.sqrt(2.0*x) * f1_vals))
    I2 = np.sum(w * ((2.0*x) * f2_vals))
    return float(I0), float(I1), float(I2)

# ------------------------------------
# Moments from the integrals
# ------------------------------------

def moments_from_integrals(I0, I1, I2, a_vec, m_t_vec, tau_t):
    """
    Compute mu^J (2-vector) and tr(Cov) from integrals and data.
      ratio_mu = I1 / I0
      ratio_v  = I2 / I0
      mu^J     = m_t + sqrt(tau_t) * (a/||a||) * ratio_mu
      tr(Cov)  = tau_t * ratio_v - ||delta_mu||^2
    with symmetry when ||a||=0.
    """
    a_norm = np.linalg.norm(a_vec)
    if I0 <= 0.0 or not np.isfinite(I0):
        # Degenerate; return NaNs to flag issues
        return np.array([np.nan, np.nan], dtype=np.float64), np.nan

    ratio_mu = I1 / I0
    ratio_v  = I2 / I0

    if a_norm == 0.0:
        delta_mu = np.array([0.0, 0.0], dtype=np.float64)
    else:
        a_hat = a_vec / a_norm
        delta_mu = math.sqrt(tau_t) * ratio_mu * a_hat

    muJ = m_t_vec + delta_mu
    trC = tau_t * ratio_v - float(np.dot(delta_mu, delta_mu))
    return muJ, trC

# ------------------------------------
# Test harness
# ------------------------------------

def make_test_vectors(num=10, max_norm=6.0, seed=123):
    """
    Construct a list of (a_vec, a_norm) with log-spaced norms in [1e-6, max_norm],
    plus an exact zero vector. Directions random on S^1.
    """
    rs = np.logspace(-6, math.log10(max_norm), num=num)
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2.0*np.pi, size=num)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # [num,2]
    a_list = [np.array([0.0, 0.0], dtype=np.float64)]  # include exact zero
    a_list += [ (r * dirs[i]).astype(np.float64) for i, r in enumerate(rs) ]
    return a_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.3, help="tau_t")
    ap.add_argument("--N", type=int, default=64, help="Gauss–Laguerre nodes for production path")
    ap.add_argument("--tol", type=float, default=1e-10, help="absolute tol for adaptive Simpson")
    args = ap.parse_args()

    tau_t = float(args.tau)
    m_t = np.array([0.0, 0.0], dtype=np.float64)  # adjust if you like

    a_vecs = make_test_vectors(num=10, max_norm=6.0, seed=123)

    mu_err_abs = []
    mu_err_rel = []
    tr_err_abs = []
    tr_err_rel = []

    print(f"Testing {len(a_vecs)} a_t values (including 0), tau={tau_t}, N={args.N}, tol={args.tol}")
    print("-"*72)

    for idx, a in enumerate(a_vecs):
        a_norm = float(np.linalg.norm(a))

        # Reference integrals (robust)
        I0_ref, I1_ref, I2_ref = integrate_reference(a_norm, tol=args.tol)
        mu_ref, tr_ref = moments_from_integrals(I0_ref, I1_ref, I2_ref, a, m_t, tau_t)

        # Production integrals (GL N small, stable)
        I0_gl, I1_gl, I2_gl = integrate_gl64(a_norm, N=args.N)
        mu_gl, tr_gl = moments_from_integrals(I0_gl, I1_gl, I2_gl, a, m_t, tau_t)

        # Errors
        mu_abs = np.linalg.norm(mu_gl - mu_ref)
        mu_ref_n = max(1e-14, np.linalg.norm(mu_ref))
        mu_rel = mu_abs / mu_ref_n

        tr_abs = abs(tr_gl - tr_ref)
        tr_ref_a = max(1e-14, abs(tr_ref))
        tr_rel = tr_abs / tr_ref_a

        mu_err_abs.append(mu_abs)
        mu_err_rel.append(mu_rel)
        tr_err_abs.append(tr_abs)
        tr_err_rel.append(tr_rel)

        print(f"#{idx:02d} ||a||={a_norm: .3e} | mu_abs={mu_abs: .3e} (rel {mu_rel: .3e}) | tr_abs={tr_abs: .3e} (rel {tr_rel: .3e})")

    mu_err_abs = np.array(mu_err_abs)
    mu_err_rel = np.array(mu_err_rel)
    tr_err_abs = np.array(tr_err_abs)
    tr_err_rel = np.array(tr_err_rel)

    def summary(name, x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"{name}: all NaN/Inf")
            return
        print(f"{name}: mean={x.mean():.3e} | median={np.median(x):.3e} | max={x.max():.3e}")

    print("\n=== Summary vs robust adaptive Simpson reference ===")
    summary("mu_abs", mu_err_abs)
    summary("mu_rel", mu_err_rel)
    summary("tr_abs", tr_err_abs)
    summary("tr_rel", tr_err_rel)

if __name__ == "__main__":
    main()
