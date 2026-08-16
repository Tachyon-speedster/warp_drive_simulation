"""
christoffel.py — Christoffel-Symbol Engine (project doc, section 4)

MATH
----
The Levi-Civita connection coefficients (Christoffel symbols of the second
kind) are:

    Gamma^a_{bc} = 1/2 * g^{ad} * ( d_b g_{dc} + d_c g_{db} - d_d g_{bc} )

where g^{ad} is the inverse metric (section 2) and d_b = d/dx^b.

Symmetry: Gamma^a_{bc} = Gamma^a_{cb} (torsion-free connection) — this is
checked explicitly in validation.py.

Storage convention: Gamma[a, b, c] = Gamma^a_{bc}, shape (4,4,4).
"""
from __future__ import annotations
import jax.numpy as jnp

from .metric import WarpBubbleParams
from .derivatives import (
    metric_and_first_derivative_autodiff,
    metric_and_first_derivative_fd,
)


def inverse_metric(g):
    """Section 2: inverse metric g^{ab}, with the identity check
    g @ g_inv == I performed separately in validation.py."""
    return jnp.linalg.inv(g)


def christoffel_symbols(g, g_inv, dg):
    """Build Gamma^a_{bc} from g, g^{-1}, and dg[d,m,n] = d_d g_{mn}.

    Gamma^a_{bc} = 1/2 g^{ad} (dg[b,d,c] + dg[c,d,b] - dg[d,b,c])
    """
    # term[b,c,d] = dg[b,d,c] + dg[c,d,b] - dg[d,b,c]
    term = (
        jnp.einsum("bdc->bcd", dg)
        + jnp.einsum("cdb->bcd", dg)
        - jnp.einsum("dbc->bcd", dg)
    )
    Gamma = 0.5 * jnp.einsum("ad,bcd->abc", g_inv, term)
    return Gamma


def christoffel_at_point(coords, params: WarpBubbleParams, engine="autodiff", h=1e-5):
    """Convenience: compute (g, g_inv, Gamma) at a single spacetime point
    using either the 'autodiff' or 'fd' derivative engine."""
    coords = jnp.asarray(coords, dtype=jnp.float64)
    if engine == "autodiff":
        g, dg = metric_and_first_derivative_autodiff(coords, params)
    elif engine == "fd":
        g, dg = metric_and_first_derivative_fd(coords, params, h=h)
    else:
        raise ValueError(f"unknown engine {engine!r}, expected 'autodiff' or 'fd'")
    g_inv = inverse_metric(g)
    Gamma = christoffel_symbols(g, g_inv, dg)
    return g, g_inv, Gamma
