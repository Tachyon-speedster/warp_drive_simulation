"""
metric.py — Spacetime Geometry Engine (project doc, section 1)

Implements the Alcubierre warp-bubble metric as a 4x4 tensor field over
coordinates x^mu = (t, x, y, z), signature (-,+,+,+), geometrized units
G = c = 1.

MATH
----
Bubble center trajectory along x:      x_s(t)
Bubble (coordinate) velocity:          v_s(t) = d(x_s)/dt
Distance from bubble center:           r_s(t,x,y,z) = sqrt((x - x_s(t))^2 + y^2 + z^2)

Shape function (top-hat smoothed by tanh, Alcubierre 1994 / Natario-style
regularized version):

    f(r_s) = [ tanh( sigma*(r_s + R) ) - tanh( sigma*(r_s - R) ) ]
             ---------------------------------------------------
                          2 * tanh( sigma * R )

Properties, all load-bearing for the rest of the pipeline:
    f(0)        -> 1   (deep interior of the bubble: flat, co-moving frame)
    f(r_s->inf) -> 0   (far away: flat, static frame / ordinary Minkowski)
    R           = bubble radius
    sigma       = wall steepness (larger sigma = thinner wall)

Line element (Alcubierre 1994):

    ds^2 = -dt^2 + (dx - v_s(t) f(r_s) dt)^2 + dy^2 + dz^2

Expanding the square gives the metric components used below:

    g_tt = -(1 - v_s^2 f^2)
    g_tx = g_xt = -v_s f
    g_xx = 1
    g_yy = 1
    g_zz = 1
    (all other components zero)

This is exactly the ADM "3+1" form with:
    lapse alpha = 1
    shift  beta^x = -v_s f,  beta^y = beta^z = 0
    spatial metric gamma_ij = delta_ij (flat spatial slices)
See section 14 (3+1 view) for how this is exploited later.

WHY THIS FORM
-------------
Every quantity downstream (Christoffel symbols, Riemann, Ricci, Einstein,
stress-energy) is generated automatically from this single function via
automatic differentiation. To change the physics of the simulation
(different bubble profile, different trajectory, a Natario-type divergence-
free shift, etc.) a user only needs to edit `shape_function` and/or
`bubble_center_x` below — nothing else in the codebase needs to change,
because christoffel.py / curvature.py only ever call `metric_tensor`.
"""
from __future__ import annotations
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class WarpBubbleParams:
    """Physical parameters of the Alcubierre bubble.

    v_s     : bubble coordinate velocity (in units of c). v_s > 1 is the
              "superluminal" warp regime; the metric is well-defined for
              any real v_s because the coordinate speed of the bubble wall
              is not a local signal speed (nothing exceeds c locally).
    R       : bubble radius (geometrized length units).
    sigma   : wall thickness parameter. Larger sigma -> thinner wall,
              steeper gradients, and (physically) larger curvature /
              larger negative energy requirement.
    x_s0    : bubble center x-position at t=0.
    """
    v_s: float = 2.0
    R: float = 1.0
    sigma: float = 8.0
    x_s0: float = 0.0

    def bubble_center_x(self, t):
        """x_s(t) = x_s0 + v_s * t  (constant-velocity trajectory)."""
        return self.x_s0 + self.v_s * t

    def bubble_velocity(self, t):
        """dx_s/dt. Constant here, but written as a function of t so that
        non-uniform trajectories (acceleration, oscillation) can be dropped
        in later without touching anything downstream."""
        return jnp.asarray(self.v_s)


def shape_function(r_s, R, sigma):
    """Alcubierre top-hat shape function f(r_s).

    f -> 1 inside the bubble (r_s << R), f -> 0 far outside (r_s >> R),
    smooth tanh transition of width ~ 1/sigma centered on r_s = R.
    """
    num = jnp.tanh(sigma * (r_s + R)) - jnp.tanh(sigma * (r_s - R))
    den = 2.0 * jnp.tanh(sigma * R)
    return num / den


def r_s_coordinate(t, x, y, z, params: WarpBubbleParams):
    """Distance from the bubble center at coordinate time t.

    NUMERICAL NOTE: sqrt(s) is not differentiable at s=0 (its derivative
    is 1/(2*sqrt(s)), which diverges), so d(r_s)/dx blows up if we ever
    evaluate exactly at the bubble center (x,y,z) == (x_s,0,0). This is a
    coordinate-singularity of the *chosen radial parametrization*, not a
    physical curvature singularity (the Alcubierre metric itself is smooth
    at r_s=0, since f'(0)=0 by symmetry). We regularize with a tiny
    epsilon inside the sqrt so autodiff never sees an exact 0/0; this
    changes r_s by at most ~1e-16 anywhere with any physical relevance and
    only meaningfully affects points within ~1e-8 of the exact center."""
    x_s = params.bubble_center_x(t)
    sum_sq = (x - x_s) ** 2 + y ** 2 + z ** 2
    eps = 1e-30
    return jnp.sqrt(sum_sq + eps)


def metric_tensor(coords, params: WarpBubbleParams):
    """Return g_{mu nu} as a (4,4) jnp array at spacetime point `coords`.

    coords: array-like [t, x, y, z]
    Index convention throughout the whole project: 0=t, 1=x, 2=y, 3=z.
    """
    t, x, y, z = coords[0], coords[1], coords[2], coords[3]
    v_s = params.bubble_velocity(t)
    r_s = r_s_coordinate(t, x, y, z, params)
    f = shape_function(r_s, params.R, params.sigma)

    g = jnp.zeros((4, 4), dtype=coords.dtype)
    g = g.at[0, 0].set(-(1.0 - v_s ** 2 * f ** 2))
    g = g.at[0, 1].set(-v_s * f)
    g = g.at[1, 0].set(-v_s * f)
    g = g.at[1, 1].set(1.0)
    g = g.at[2, 2].set(1.0)
    g = g.at[3, 3].set(1.0)
    return g


def minkowski_metric(dtype=jnp.float64):
    """eta_{mu nu} = diag(-1,1,1,1). Used for the flat-spacetime validation
    limit (section 19) and as v_s -> 0 sanity check."""
    return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=dtype))
