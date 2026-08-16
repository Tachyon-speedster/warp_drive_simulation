"""
observer.py — Observer / Four-Velocity System & Proper Time
(project doc, sections 10 & 11)

MATH
----
A four-velocity u^a is the tangent vector to an observer's worldline,
parameterized by the observer's own proper time tau:  u^a = dx^a/dtau.

Normalization condition for a physical, massive (timelike) observer:

    g_{ab} u^a u^b = -1     (signature -,+,+,+)

Given an *unnormalized* candidate spatial 3-velocity direction (or a raw
4-vector), we normalize it against the local metric so that the above
holds. For a static observer sitting at fixed (x,y,z) with only a time
component, u^a = (u^t, 0,0,0), normalization reduces to:

    g_{tt} (u^t)^2 = -1   =>   u^t = 1/sqrt(-g_tt)

which is exactly the standard "static observer" construction, generalized
here to allow an arbitrary spatial direction n^i via

    u^a = Gamma_L * (1, v n^x, v n^y, v n^z)

solved for the normalization factor Gamma_L such that g_{ab}u^a u^b = -1,
i.e. a local Lorentz-factor-like rescaling with respect to the *curved*
metric at that point (not flat Minkowski).

Proper time relation:

    dtau^2 = -g_{ab} dx^a dx^b        (for a timelike interval)

so that along any candidate displacement dx^a,
    dtau = sqrt( -g_{ab} dx^a dx^b )
provided the interval is timelike (the expression under the sqrt is > 0).
We raise an error otherwise — enforced by `is_timelike` below, matching the
project rule in section 10: "reject vectors that are not timelike when a
timelike observer is required."
"""
from __future__ import annotations
import jax.numpy as jnp

from .adm import adm_decompose


def norm_squared(g, u):
    """g_{ab} u^a u^b, a scalar. Negative => timelike, zero => null,
    positive => spacelike (signature -,+,+,+)."""
    return jnp.einsum("ab,a,b->", g, u, u)


def is_timelike(g, u, tol=1e-9):
    return bool(norm_squared(g, u) < -tol)


def is_null(g, u, tol=1e-9):
    return bool(jnp.abs(norm_squared(g, u)) < tol)


def normalize_static_observer(g):
    """u^a = (1/sqrt(-g_tt), 0, 0, 0): an observer instantaneously at rest
    in the coordinate spatial grid (NOT necessarily co-moving with the warp
    bubble). Requires g_tt < 0 at this point.

    IMPORTANT PHYSICAL CAVEAT: for a *superluminal* bubble (v_s > 1),
    g_tt = -(1 - v_s^2 f^2) becomes POSITIVE deep inside the bubble
    (f -> 1), because the coordinate-static worldline dx^i=0 is genuinely
    spacelike there -- the bubble interior is being carried along faster
    than a fixed-grid observer can "keep up" with in this coordinate
    sense. This is a real, well-known feature of the Alcubierre solution,
    not a bug. Use `normalize_eulerian_observer` for a choice of observer
    that is guaranteed timelike everywhere (see below)."""
    g_tt = g[0, 0]
    if g_tt >= 0:
        raise ValueError(
            "g_tt >= 0 at this point: a coordinate-static observer is not "
            "timelike here. For v_s > 1 (superluminal bubble) this happens "
            "inside the bubble by construction -- use "
            "normalize_eulerian_observer() instead, which is timelike "
            "everywhere the ADM lapse is well-defined."
        )
    u_t = 1.0 / jnp.sqrt(-g_tt)
    u = jnp.array([u_t, 0.0, 0.0, 0.0], dtype=g.dtype)
    if not is_timelike(g, u):
        raise ValueError("Constructed static observer failed timelike check.")
    return u


def normalize_eulerian_observer(g, check=True):
    """The ADM "normal" (Eulerian) observer: the observer whose worldline
    is always orthogonal to the constant-t spatial slices. In terms of the
    3+1 (ADM) split (adm.py, section 14):

        n^a = (1/alpha) * (1, -beta^1, -beta^2, -beta^3)

    This four-velocity satisfies g_ab n^a n^b = -alpha^2 * (1/alpha)^2 = -1
    EXACTLY and ALWAYS, for any valid ADM decomposition with alpha > 0 --
    it never needs a "is g_tt negative" caveat the way the naive
    coordinate-static observer does. For the Alcubierre metric, alpha = 1
    identically (metric.py), so n^a = (1, v_s*f, 0, 0): physically, this is
    the observer who free-falls along with the local warp of space (the
    natural "co-moving with the geometry" observer), which is exactly why
    it stays timelike even deep inside a superluminal bubble where the
    coordinate-static observer does not.

    This is used as the DEFAULT observer for energy-density field maps
    (grid.py) precisely because it is always well-defined."""
    alpha, beta_i, beta_up, gamma_ij = adm_decompose(g)
    u = jnp.concatenate([jnp.array([1.0], dtype=g.dtype), -beta_up]) / alpha
    if check:
        # NOTE: `check=True` triggers a Python-level bool() conversion,
        # which only works for concrete (non-traced) arrays -- i.e. eager
        # single-point calls. Any call made under jax.vmap/jax.jit (e.g.
        # the grid field evaluator in grid.py) MUST pass check=False,
        # since a traced boolean cannot be branched on in Python.
        if not is_timelike(g, u):
            raise ValueError(
                "Eulerian observer failed timelike check -- the ADM lapse "
                "is likely ill-defined (alpha^2 <= 0) at this point."
            )
    return u


def normalize_moving_observer(g, spatial_velocity):
    """Build a normalized timelike four-velocity for an observer moving
    with local 3-velocity `spatial_velocity` = (vx, vy, vz) (coordinate
    velocity, not necessarily < 1 in flat-space sense since the metric is
    curved). We solve for the overall scale k such that
    u = k * (1, vx, vy, vz) satisfies g_ab u^a u^b = -1:

        k^2 * [ g_tt + 2 g_ti v^i + g_ij v^i v^j ] = -1
        =>  k = 1 / sqrt( -(g_tt + 2 g_ti v^i + g_ij v^i v^j) )
    """
    vx, vy, vz = spatial_velocity
    dir_vec = jnp.array([1.0, vx, vy, vz], dtype=g.dtype)
    quad = jnp.einsum("ab,a,b->", g, dir_vec, dir_vec)
    if quad >= 0:
        raise ValueError(
            "Requested spatial_velocity is not timelike at this point "
            "(g_ab dir^a dir^b >= 0); reduce |v| or choose another point."
        )
    k = 1.0 / jnp.sqrt(-quad)
    u = k * dir_vec
    if not is_timelike(g, u):
        raise ValueError("Constructed moving observer failed timelike check.")
    return u


def proper_time_interval(g, dx):
    """dtau = sqrt(-g_ab dx^a dx^b) for a timelike displacement dx^a.
    Raises if the interval is not timelike."""
    quad = jnp.einsum("ab,a,b->", g, dx, dx)
    if quad >= 0:
        raise ValueError(
            "Displacement is not timelike (ds^2 >= 0); proper time is "
            "undefined for null/spacelike separations."
        )
    return jnp.sqrt(-quad)


def energy_density(T, u):
    """Observer-measured energy density: rho = T_{ab} u^a u^b.
    This is the quantity whose sign directly tests the energy conditions
    (section 20 / milestone 7-9): rho < 0 for ANY timelike observer at a
    point signals violation of the Weak Energy Condition there."""
    return jnp.einsum("ab,a,b->", T, u, u)
