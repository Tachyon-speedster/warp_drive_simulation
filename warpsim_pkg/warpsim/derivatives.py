"""
derivatives.py — Metric Derivative Engine (project doc, sections 3 & 18)

Two independent engines compute d(g_{mu nu}) / d(x^alpha), a rank-3 tensor
we store as an array `dg` with shape (4,4,4) and index convention

    dg[a, m, n] = d g_{mn} / d x^a

AUTODIFF ENGINE (primary, section 18)
--------------------------------------
Uses jax.jacfwd on `metric_tensor`. This is exact to machine precision
(no discretization/truncation error at all — it is literally the analytic
derivative evaluated via the chain rule through the tanh/sqrt expressions),
which matters most right at the warp-bubble wall where finite differences
are least trustworthy (large second derivatives of f -> the classic
"catastrophic cancellation near a steep transition" failure mode of finite
differencing). It is also reused *again* (nested jacfwd) to get the second
metric derivatives needed for the Riemann tensor (curvature.py), which
finite differences handle very poorly (error grows as the square of the
first-derivative error).

FINITE-DIFFERENCE ENGINE (legacy, retained per project rule in section 3 /
section 18: "the old implementation must remain available for comparison")
----------------------------------------------------------------------------
Central differences, O(h^2) accurate:

    d g_{mn}/dx^a  ~=  [ g_{mn}(x + h*e_a) - g_{mn}(x - h*e_a) ] / (2h)

Second derivatives (needed only by the FD Riemann path) use the standard
mixed central-difference stencil.

Both engines expose the SAME call signature so curvature.py / christoffel.py
can be pointed at either one via a single string flag ("autodiff" | "fd").
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from .metric import metric_tensor, WarpBubbleParams

# ---------------------------------------------------------------------------
# AUTODIFF ENGINE
# ---------------------------------------------------------------------------

def metric_and_first_derivative_autodiff(coords, params: WarpBubbleParams):
    """Returns (g, dg) where dg[a,m,n] = d g_{mn} / d x^a, computed exactly
    via forward-mode automatic differentiation."""
    g = metric_tensor(coords, params)
    # jacfwd differentiates w.r.t. `coords` (shape (4,)) of a (4,4)-valued
    # function -> jacobian has shape (4,4,4) with axes (m, n, a). We move
    # the derivative axis `a` to the front to match our dg[a,m,n] convention.
    jac = jax.jacfwd(lambda c: metric_tensor(c, params))(coords)  # (m,n,a)
    dg = jnp.moveaxis(jac, -1, 0)  # (a,m,n)
    return g, dg


def second_derivative_autodiff(coords, params: WarpBubbleParams):
    """Returns d2g[a,b,m,n] = d^2 g_{mn} / (dx^a dx^b), exact via nested
    forward-mode autodiff (jacfwd of jacfwd). This is the operation that is
    numerically fragile under finite differencing but essentially free
    (still exact, still one extra pass) under autodiff."""
    def g_of_c(c):
        return metric_tensor(c, params)

    hess = jax.jacfwd(jax.jacfwd(g_of_c))(coords)  # shape (m,n,a,b)
    d2g = jnp.moveaxis(hess, (-2, -1), (0, 1))  # -> (a,b,m,n)
    return d2g


# ---------------------------------------------------------------------------
# FINITE-DIFFERENCE ENGINE (legacy — kept for validation/comparison)
# ---------------------------------------------------------------------------

def metric_and_first_derivative_fd(coords, params: WarpBubbleParams, h=1e-5):
    """Central-difference first derivative, O(h^2) truncation error."""
    coords = jnp.asarray(coords, dtype=jnp.float64)
    g = metric_tensor(coords, params)
    dg = np.zeros((4, 4, 4))
    for a in range(4):
        step = np.zeros(4)
        step[a] = h
        c_plus = coords + jnp.asarray(step)
        c_minus = coords - jnp.asarray(step)
        g_plus = np.asarray(metric_tensor(c_plus, params))
        g_minus = np.asarray(metric_tensor(c_minus, params))
        dg[a] = (g_plus - g_minus) / (2 * h)
    return g, jnp.asarray(dg)


def second_derivative_fd(coords, params: WarpBubbleParams, h=1e-4):
    """Central-difference second (mixed) derivative, O(h^2) truncation
    error. Uses the standard 4-point stencil for d^2/(dx^a dx^b), a>=b,
    and symmetry d2g[a,b]=d2g[b,a] to fill the rest."""
    coords = np.asarray(coords, dtype=np.float64)
    d2g = np.zeros((4, 4, 4, 4))

    def g_at(c):
        return np.asarray(metric_tensor(jnp.asarray(c), params))

    for a in range(4):
        for b in range(a, 4):
            ea = np.zeros(4); ea[a] = h
            eb = np.zeros(4); eb[b] = h
            if a == b:
                g_pp = g_at(coords + ea)
                g_0 = g_at(coords)
                g_mm = g_at(coords - ea)
                d2 = (g_pp - 2 * g_0 + g_mm) / (h ** 2)
            else:
                g_pp = g_at(coords + ea + eb)
                g_pm = g_at(coords + ea - eb)
                g_mp = g_at(coords - ea + eb)
                g_mm = g_at(coords - ea - eb)
                d2 = (g_pp - g_pm - g_mp + g_mm) / (4 * h ** 2)
            d2g[a, b] = d2
            d2g[b, a] = d2
    return jnp.asarray(d2g)


def compare_engines(coords, params: WarpBubbleParams, h=1e-5):
    """Diagnostic used by validation.py (section 19): returns the max
    absolute and max relative discrepancy between the autodiff and
    finite-difference first derivatives at a point. Large discrepancies are
    expected exactly at/inside the wall region where FD truncation +
    round-off both blow up."""
    coords_j = jnp.asarray(coords, dtype=jnp.float64)
    _, dg_ad = metric_and_first_derivative_autodiff(coords_j, params)
    _, dg_fd = metric_and_first_derivative_fd(coords_j, params, h=h)
    diff = np.asarray(dg_ad) - np.asarray(dg_fd)
    max_abs = np.max(np.abs(diff))
    denom = np.maximum(np.abs(np.asarray(dg_ad)), 1e-12)
    max_rel = np.max(np.abs(diff) / denom)
    return {"max_abs_diff": float(max_abs), "max_rel_diff": float(max_rel)}
