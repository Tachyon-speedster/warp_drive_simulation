"""
curvature.py — Riemann / Ricci / Einstein Tensor Engines
(project doc, sections 5, 6, 7, 8)

MATH
----
Riemann tensor (mixed form, R^a_{bcd}):

    R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
                + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}

This requires the *derivative of the Christoffel symbols*, i.e. effectively
second derivatives of the metric. This is exactly the operation the project
doc (sections 3, 18) flags as the reason to move off finite differences: the
FD path here differentiates an already-FD-differentiated quantity, roughly
squaring the truncation/round-off error. The autodiff path instead
differentiates the *exact analytic* Christoffel-symbol function directly
(jax.jacfwd of a function that is itself built from jax.jacfwd), so no error
compounding occurs — the result is exact to machine (float64) precision
regardless of how sharp the warp wall is.

Ricci tensor (contraction of Riemann on 1st & 3rd index):

    R_{bd} = R^a_{bad}   (sum over a)

Ricci scalar:

    R = g^{bd} R_{bd}

Einstein tensor:

    G_{ab} = R_{ab} - 1/2 g_{ab} R

Symmetries used as validation checks (section 19):
    R^a_{bcd} = -R^a_{bdc}                      (antisymmetry in last 2)
    R_{bd} = R_{db}                             (Ricci symmetry)
    G_{ab} = G_{ba}                             (Einstein tensor symmetry)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from .metric import WarpBubbleParams, metric_tensor
from .derivatives import metric_and_first_derivative_autodiff, metric_and_first_derivative_fd
from .christoffel import inverse_metric, christoffel_symbols


# ---------------------------------------------------------------------------
# Build Gamma(coords) as a pure function of coords, so it can itself be
# differentiated (autodiff engine) or finite-differenced (FD engine).
# ---------------------------------------------------------------------------

def _gamma_function_autodiff(params: WarpBubbleParams):
    def gamma_of_coords(coords):
        g, dg = metric_and_first_derivative_autodiff(coords, params)
        g_inv = inverse_metric(g)
        return christoffel_symbols(g, g_inv, dg)
    return gamma_of_coords


def _gamma_function_fd(params: WarpBubbleParams, h_inner):
    def gamma_of_coords(coords):
        g, dg = metric_and_first_derivative_fd(coords, params, h=h_inner)
        g_inv = inverse_metric(g)
        return christoffel_symbols(g, g_inv, dg)
    return gamma_of_coords


def _dgamma_fd(gamma_func, coords, h_outer):
    """Central-difference derivative of Gamma^a_{bc} w.r.t. x^d, returned
    as dGamma[d,a,b,c]. Used only by the FD engine (two nested FD passes —
    this is the fragile, legacy path)."""
    coords = np.asarray(coords, dtype=np.float64)
    dGamma = np.zeros((4, 4, 4, 4))
    for d in range(4):
        step = np.zeros(4); step[d] = h_outer
        g_plus = np.asarray(gamma_func(jnp.asarray(coords + step)))
        g_minus = np.asarray(gamma_func(jnp.asarray(coords - step)))
        dGamma[d] = (g_plus - g_minus) / (2 * h_outer)
    return jnp.asarray(dGamma)


def riemann_tensor(coords, params: WarpBubbleParams, engine="autodiff",
                    h_outer=1e-4, h_inner=1e-4):
    """Returns (g, g_inv, Gamma, Riemann) at a point.

    Riemann[a,b,c,d] = R^a_{bcd}
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)

    if engine == "autodiff":
        gamma_func = _gamma_function_autodiff(params)
        Gamma = gamma_func(coords)
        # jacfwd of a (4,4,4)-valued function w.r.t. a (4,)-valued input
        # gives shape (a,b,c,d_deriv); move deriv axis to front -> dGamma[d,a,b,c]
        jac = jax.jacfwd(gamma_func)(coords)  # (a,b,c,d)
        dGamma = jnp.moveaxis(jac, -1, 0)  # (d,a,b,c)
    elif engine == "fd":
        gamma_func = _gamma_function_fd(params, h_inner)
        Gamma = gamma_func(coords)
        dGamma = _dgamma_fd(gamma_func, coords, h_outer)
    else:
        raise ValueError(f"unknown engine {engine!r}")

    g = metric_tensor(coords, params)
    g_inv = inverse_metric(g)

    # R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
    #             + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}
    term1 = jnp.einsum("cabd->abcd", dGamma)  # d_c Gamma^a_{bd}
    term2 = jnp.einsum("dabc->abcd", dGamma)  # d_d Gamma^a_{bc}
    term3 = jnp.einsum("ace,ebd->abcd", Gamma, Gamma)
    term4 = jnp.einsum("ade,ebc->abcd", Gamma, Gamma)
    Riemann = term1 - term2 + term3 - term4
    return g, g_inv, Gamma, Riemann


def ricci_tensor(Riemann):
    """R_{bd} = R^a_{bad}  (contract 1st & 3rd index of the mixed Riemann)."""
    return jnp.einsum("abad->bd", Riemann)


def ricci_scalar(g_inv, Ric):
    """R = g^{bd} R_{bd}."""
    return jnp.einsum("bd,bd->", g_inv, Ric)


def einstein_tensor(g, g_inv, Ric, R_scalar):
    """G_{ab} = R_{ab} - 1/2 g_{ab} R."""
    return Ric - 0.5 * g * R_scalar


def full_curvature_at_point(coords, params: WarpBubbleParams, engine="autodiff",
                             h_outer=1e-4, h_inner=1e-4):
    """One-stop call returning every geometric quantity at a point:
    g, g_inv, Gamma, Riemann, Ricci, R (scalar), Einstein tensor."""
    g, g_inv, Gamma, Riemann = riemann_tensor(
        coords, params, engine=engine, h_outer=h_outer, h_inner=h_inner
    )
    Ric = ricci_tensor(Riemann)
    R_scalar = ricci_scalar(g_inv, Ric)
    G = einstein_tensor(g, g_inv, Ric, R_scalar)
    return {
        "g": g, "g_inv": g_inv, "Gamma": Gamma, "Riemann": Riemann,
        "Ricci": Ric, "R_scalar": R_scalar, "Einstein": G,
    }
