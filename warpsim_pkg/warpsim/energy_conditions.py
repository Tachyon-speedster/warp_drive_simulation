"""
energy_conditions.py — Energy Conditions (project doc "Next" milestones 6-10)

MATH
----
Given the local stress-energy tensor T_ab and a timelike unit observer u^a
(g_ab u^a u^b = -1), three standard energy conditions:

WEC (Weak Energy Condition):
    T_ab u^a u^b >= 0   for every timelike u^a.
    This is exactly `observer.energy_density`; a negative value at any
    point, for any observer, is already a WEC violation there.

NEC (Null Energy Condition):
    T_ab k^a k^b >= 0   for every NULL k^a (g_ab k^a k^b = 0).
    NEC is *weaker* than WEC (it's implied by WEC via a continuity/limiting
    argument: null vectors are limits of timelike vectors), so NEC
    violation is the more fundamental, harder-to-avoid statement — it is
    what's actually used to prove exotic matter is unavoidable for warp
    drives and traversable wormholes (Morris-Thorne / Alcubierre results).

DEC (Dominant Energy Condition):
    WEC, AND the local energy-flux vector q^a = -T^a_b u^b must be
    non-spacelike (g_ab q^a q^b <= 0), i.e. energy cannot flow faster than
    light in any observer's frame.

CONSTRUCTING TEST NULL VECTORS
-------------------------------
Given a timelike unit observer u^a, any vector of the form

    k^a(theta) = u^a + cos(theta) e1^a + sin(theta) e2^a

is automatically null, where {e1,e2} are spacelike unit vectors orthogonal
to u^a AND to each other (g_ab u^a e_i^b = 0, g_ab e_i^a e_j^b = delta_ij):

    g_ab k^a k^b = g_ab u^a u^b + 2cos(theta) g_ab u^a e1^b
                   + 2sin(theta) g_ab u^a e2^b + cos^2 g_ab e1 e1
                   + sin^2 g_ab e2 e2 + 2 sin*cos g_ab e1 e2
                 = -1 + 0 + 0 + cos^2(theta) + sin^2(theta) + 0 = 0.

{e1,e2,e3} (a full orthonormal spatial triad orthogonal to u) are built by
Gram-Schmidt on the coordinate basis vectors d/dx, d/dy, d/dz, projected
orthogonal to u^a first. This module restricts sampling to the {e1,e2}
(x-y) plane, matching the 2D (z=0) grid used throughout the project, and
sweeps `theta` over many directions to approximate "for every null k^a."
NEC checked this way is a numerical *sample*, not an analytic proof — if
you need a guarantee, the true minimum over theta can be found by root-
finding on d/dtheta[T_ab k^a k^b] = 0 instead of a fixed grid of angles;
the fixed-angle sweep is a good and fast diagnostic in practice because
T_ab k^a k^b is a smooth low-order trig polynomial in theta.
"""
from __future__ import annotations
import jax.numpy as jnp


def orthonormal_spatial_frame(g, u):
    """Gram-Schmidt orthonormal spatial triad {e1,e2,e3} orthogonal to the
    observer u^a, w.r.t. the metric g. Returns a (3,4) array, rows e1,e2,e3."""
    coord_basis = [
        jnp.array([0.0, 1.0, 0.0, 0.0], dtype=g.dtype),
        jnp.array([0.0, 0.0, 1.0, 0.0], dtype=g.dtype),
        jnp.array([0.0, 0.0, 0.0, 1.0], dtype=g.dtype),
    ]
    frame = []
    for b in coord_basis:
        # project out the u-component: since g(u,u)=-1,
        # v_perp = b - (g(u,b)/g(u,u)) u = b + g(u,b) u
        gub = jnp.einsum("ab,a,b->", g, u, b)
        v = b + gub * u
        # Gram-Schmidt against previously built (already orthonormal) e_i
        for e in frame:
            coeff = jnp.einsum("ab,a,b->", g, e, v)
            v = v - coeff * e
        norm = jnp.sqrt(jnp.einsum("ab,a,b->", g, v, v))
        frame.append(v / norm)
    return jnp.stack(frame, axis=0)  # (3,4)


def null_vectors_in_plane(g, u, n_theta=16):
    """Sample n_theta null vectors k(theta) = u + cos(theta) e1 + sin(theta) e2
    in the spatial x-y plane orthogonal to u. Returns array (n_theta, 4)."""
    frame = orthonormal_spatial_frame(g, u)
    e1, e2 = frame[0], frame[1]
    thetas = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False)
    k = u[None, :] + jnp.cos(thetas)[:, None] * e1[None, :] + jnp.sin(thetas)[:, None] * e2[None, :]
    return k


def check_wec(T, u):
    """WEC: T_ab u^a u^b >= 0. Returns (rho, passes)."""
    rho = jnp.einsum("ab,a,b->", T, u, u)
    return rho, rho >= 0.0


def check_nec(T, g, u, n_theta=16):
    """NEC sampled over n_theta null directions in the local x-y plane.
    Returns (min_value, passes) where min_value = min_theta T_ab k^a k^b."""
    k = null_vectors_in_plane(g, u, n_theta=n_theta)
    vals = jnp.einsum("ab,na,nb->n", T, k, k)
    min_val = jnp.min(vals)
    return min_val, min_val >= 0.0


def check_dec(T, g, u):
    """DEC: WEC holds AND the energy-flux vector q^a = -T^a_b u^b is
    non-spacelike (g_ab q^a q^b <= 0). Returns (rho, flux_norm_sq, passes)."""
    rho, wec_ok = check_wec(T, u)
    g_inv = jnp.linalg.inv(g)
    T_mixed = jnp.einsum("ac,cb->ab", g_inv, T)  # T^a_b
    q = -jnp.einsum("ab,b->a", T_mixed, u)
    flux_norm_sq = jnp.einsum("ab,a,b->", g, q, q)
    dec_ok = jnp.logical_and(wec_ok, flux_norm_sq <= 1e-9)
    return rho, flux_norm_sq, dec_ok
