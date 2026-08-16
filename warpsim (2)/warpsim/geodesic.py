"""
geodesic.py — Geodesic Engine + Validation (project doc, sections 12 & 13)

MATH
----
The geodesic equation (freely-falling particle, no force other than
gravity/geometry itself):

    d^2 x^a / dlambda^2  +  Gamma^a_{bc} (dx^b/dlambda) (dx^c/dlambda) = 0

For a massive particle we use proper time tau as the affine parameter
lambda. This is a second-order ODE in x^a(tau); we convert it to a
first-order system in the state vector y = (x^0,x^1,x^2,x^3, u^0,u^1,u^2,u^3)
where u^a = dx^a/dtau is the four-velocity:

    dx^a/dtau = u^a
    du^a/dtau = -Gamma^a_{bc}(x) u^b u^c

and integrate with `scipy.integrate.solve_ivp` (adaptive Runge-Kutta,
RK45 by default — matching the project note that "current integration uses
SciPy's numerical ODE functionality").

VALIDATION (section 13)
------------------------
A massive particle's four-velocity must stay normalized along the whole
trajectory:  g_{ab}(x(tau)) u^a u^b = -1  for all tau. Because the geodesic
equation is *derived* from this constraint being conserved (it is a first
integral of motion for the Levi-Civita connection), a normalization error
that grows during integration is a direct, physically meaningful measure
of numerical integration error — not a modeling error. We track it at every
solver step and report max drift as `normalization_error`.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from .metric import WarpBubbleParams, metric_tensor
from .christoffel import christoffel_at_point


def _christoffel_fn_jit(params: WarpBubbleParams):
    """JIT-compiled autodiff Christoffel-symbol evaluator, reused at every
    integrator substep for speed (geodesic integration calls this
    thousands of times per trajectory)."""
    def gamma_of_coords(coords):
        _, _, Gamma = christoffel_at_point(coords, params, engine="autodiff")
        return Gamma
    return jax.jit(gamma_of_coords)


def integrate_geodesic(params: WarpBubbleParams, x0, u0, tau_span,
                        n_eval=400, rtol=1e-10, atol=1e-12,
                        method="RK45", norm_target=-1.0):
    """Integrate the geodesic equation.

    x0     : initial position [t,x,y,z]
    u0     : initial four-velocity [u^t,u^x,u^y,u^z]. For a massive
             particle this should already satisfy g_ab u^a u^b = -1 (see
             observer.py); for a light ray (null geodesic) it should
             satisfy g_ab u^a u^b = 0 (see `null_ray_direction` below).
             This function does NOT re-normalize u0 for you.
    tau_span: (lambda_start, lambda_end) -- proper time for massive
             particles, or an arbitrary affine parameter for null rays
             (proper time is not defined along a null worldline).
    norm_target: the value g_ab u^a u^b should stay conserved at along the
             true solution (-1 for massive, 0 for null). Only used to
             compute the reported normalization-drift diagnostic; it does
             not affect the integration itself, which is the same ODE
             either way.

    Returns dict with position(param), four_velocity(param), the parameter
    array, final position/velocity, and the normalization-error trace
    (section 13).
    """
    gamma_fn = _christoffel_fn_jit(params)

    def rhs(tau, y):
        x = jnp.asarray(y[0:4])
        u = jnp.asarray(y[4:8])
        Gamma = gamma_fn(x)  # Gamma[a,b,c]
        du = -jnp.einsum("abc,b,c->a", Gamma, u, u)
        return np.concatenate([np.asarray(u), np.asarray(du)])

    y0 = np.concatenate([np.asarray(x0, dtype=np.float64),
                          np.asarray(u0, dtype=np.float64)])
    t_eval = np.linspace(tau_span[0], tau_span[1], n_eval)

    sol = solve_ivp(rhs, tau_span, y0, method=method, t_eval=t_eval,
                     rtol=rtol, atol=atol, dense_output=False)

    positions = sol.y[0:4, :].T          # (n_eval, 4)
    velocities = sol.y[4:8, :].T         # (n_eval, 4)

    norm_errors = np.zeros(positions.shape[0])
    for i in range(positions.shape[0]):
        g = np.asarray(metric_tensor(jnp.asarray(positions[i]), params))
        u = velocities[i]
        norm = g @ u @ u  # g_ab u^a u^b
        norm_errors[i] = norm - norm_target

    return {
        "tau": sol.t,
        "position": positions,
        "four_velocity": velocities,
        "final_position": positions[-1],
        "final_velocity": velocities[-1],
        "normalization_error": norm_errors,
        "max_normalization_error": float(np.max(np.abs(norm_errors))),
        "success": sol.success,
        "message": sol.message,
    }


def null_ray_direction(g, spatial_direction):
    """Build a null four-velocity k^a = k^t * (1, n^x, n^y, n^z) for a light
    ray launched in coordinate spatial direction `spatial_direction`
    (need not be unit -- only its direction matters).

    Solve for k^t from g_ab k^a k^b = 0 with k^a = k^t*(1, n^i):

        (k^t)^2 * [ g_tt + 2 g_ti n^i + g_ij n^i n^j ] = 0

    This is satisfied for any k^t only if the bracket itself is zero,
    which is generically NOT the case for an arbitrary direction n^i in a
    curved metric (unlike flat space, "any direction at speed 1" is not
    automatically null once g_ti != 0). Instead we solve the correct
    quadratic in k^t directly from the null condition written as
    g_tt (k^t)^2 + 2 g_ti k^t k^i + g_ij k^i k^j = 0 with k^i = k^t n^i,
    i.e. treat n^i as fixing the RATIO k^i/k^t and solve for the physical
    requirement that such a k^a be null -- but since the equation is
    homogeneous of degree 2 in k^t, k^t itself cancels and we are left
    needing g_tt + 2 g_ti n^i + g_ij n^i n^j = 0, i.e. n^i must be chosen
    (rescaled) such that this holds. We do that by writing this as a
    quadratic in the OVERALL SCALE of the given raw direction: given a raw
    unit spatial direction e^i (in the flat coordinate sense), solve for
    the scalar s such that n^i = s*e^i satisfies the null condition:

        g_ij s^2 e^i e^j + 2 g_ti s e^i + g_tt = 0   (quadratic in s)

    and pick the positive root. This is exactly the standard
    "photon emitted in direction e, what k^t/k^i ratio is null" problem.
    """
    e = jnp.asarray(spatial_direction, dtype=g.dtype)
    e = e / jnp.linalg.norm(e)
    A = jnp.einsum("ij,i,j->", g[1:4, 1:4], e, e)
    B = 2.0 * jnp.einsum("i,i->", g[0, 1:4], e)
    C = g[0, 0]
    disc = B ** 2 - 4 * A * C
    s = (-B + jnp.sqrt(disc)) / (2 * A)
    n = s * e
    k = jnp.concatenate([jnp.array([1.0], dtype=g.dtype), n])
    return k
