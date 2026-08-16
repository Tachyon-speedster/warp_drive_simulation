"""
adm_evolution.py — Full 3+1 (ADM) Dynamical Evolution Equations
(extends adm.py, project doc section 21-23 "Advanced" milestones)

WHY THIS MODULE EXISTS
-----------------------
adm.py already builds the algebraic 3+1 split (alpha, beta^i, gamma_ij),
the extrinsic curvature K_ij, and the two ADM *constraint* equations
(Hamiltonian + momentum). Those are necessary but not sufficient to call
this a genuine "3+1 spacetime construct": constraints only check that a
single time-slice is a valid initial-data set. They say nothing about
whether the geometry actually evolves the way Einstein's equations say a
3+1 foliation must evolve from one slice to the next.

This module closes that gap by implementing the second half of the ADM
system -- the dynamical evolution equation for the extrinsic curvature --

    d/dt K_ij = beta^k D_k K_ij + K_ik D_j beta^k + K_kj D_i beta^k
                - D_i D_j alpha
                + alpha * [ (3)R_ij + K K_ij - 2 K_ik gamma^kl K_lj ]
                - 8*pi*alpha * [ S_ij - (1/2) gamma_ij (S - rho) ]

(MTW/Baumgarte-Shapiro sign convention, matching the K_ij = -Gamma^0_ij
convention already fixed in adm.py) and cross-checks it against the
*independent* autodiff time-derivative of the closed-form
K_ij(t,x,y,z) = -Gamma^0_ij(t,x,y,z) already used in adm.py.

Because the Alcubierre metric is an *exact*, closed-form solution for all
t (not just t=0), this is a much stronger test than the usual "evolve one
step and hope it's close": the two sides of the check above are computed
by completely different routes (one via the full nonlinear RHS built from
spatial curvature + shift + matter source terms, the other via a single
autodiff derivative through the exact metric with respect to t) and must
agree to machine precision at *every* spacetime point, for *any* choice of
bubble parameters, if -- and only if -- the whole pipeline (metric ->
Christoffel -> Riemann -> Ricci -> Einstein -> stress-energy -> ADM
projection) is self-consistent. This is therefore the strongest available
correctness test on the whole codebase; the Hamiltonian/momentum
constraint checks in adm.py only ever exercised half of the ADM system.

CONVENTIONS
-----------
- (3)R_ij is built from first principles here (not assumed to vanish),
  using the exact same Riemann-from-Christoffel pattern as curvature.py,
  just restricted to the spatial (3D) slice at fixed t. For the Alcubierre
  family gamma_ij = delta_ij for all t, so (3)R_ij must come out
  numerically zero -- but computing it generally (rather than hard-coding
  zero) means this module stays correct if metric.py is ever extended to
  a metric family with non-flat spatial slices (Natario-type, etc.).
- S_ij, S, rho are the matter fields measured by the Eulerian (ADM-normal)
  observer, obtained via the general tensor projector
  gamma^a_b = delta^a_b + n^a n_b (not a shortcut specific to this
  metric), so the same code is correct for any alpha, beta^i.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

from .metric import WarpBubbleParams, metric_tensor
from .christoffel import christoffel_at_point
from .curvature import full_curvature_at_point
from .stress_energy import stress_energy_tensor
from .observer import normalize_eulerian_observer, energy_density
from .adm import adm_decompose


# ---------------------------------------------------------------------------
# Spatial (3D) geometry of the t=const slice: Christoffel, Riemann, Ricci.
# Exactly mirrors christoffel.py / curvature.py's index conventions, just
# restricted to the 3x3 spatial block at fixed t.
# ---------------------------------------------------------------------------

def _spatial_metric(t, xyz, params: WarpBubbleParams):
    """gamma_ij(x,y,z) at fixed t, as a pure function of xyz (3,) so it can
    be jacfwd'ed for spatial derivatives."""
    coords = jnp.concatenate([jnp.array([t], dtype=xyz.dtype), xyz])
    g = metric_tensor(coords, params)
    return g[1:4, 1:4]


def spatial_christoffel(t, xyz, params: WarpBubbleParams):
    """(3)Gamma^k_ij of the spatial slice at (t, xyz). Returns
    (gamma_ij, gamma^ij, Gamma3[k,i,j])."""
    gamma = _spatial_metric(t, xyz, params)
    jac = jax.jacfwd(lambda x: _spatial_metric(t, x, params))(xyz)  # (m,n,d)
    dgamma = jnp.moveaxis(jac, -1, 0)  # dgamma[d,m,n] = d_d gamma_mn
    gamma_inv = jnp.linalg.inv(gamma)
    # Identical structure to christoffel_symbols() in christoffel.py, 3D.
    term = (
        jnp.einsum("bdc->bcd", dgamma)
        + jnp.einsum("cdb->bcd", dgamma)
        - jnp.einsum("dbc->bcd", dgamma)
    )
    Gamma3 = 0.5 * jnp.einsum("ad,bcd->abc", gamma_inv, term)
    return gamma, gamma_inv, Gamma3


def spatial_ricci(t, xyz, params: WarpBubbleParams):
    """(3)R_ij of the spatial slice at (t, xyz), via the same
    Riemann-from-Christoffel contraction pattern as curvature.py
    (riemann_tensor + ricci_tensor), restricted to 3D."""
    def gamma3_of_xyz(x):
        _, _, Gamma3 = spatial_christoffel(t, x, params)
        return Gamma3

    Gamma3 = gamma3_of_xyz(xyz)
    jac2 = jax.jacfwd(gamma3_of_xyz)(xyz)      # (a,b,c,d) = d_d Gamma3^a_bc
    dGamma3 = jnp.moveaxis(jac2, -1, 0)        # (d,a,b,c) = d_d Gamma3^a_bc

    term1 = jnp.einsum("cabd->abcd", dGamma3)  # d_c Gamma3^a_bd
    term2 = jnp.einsum("dabc->abcd", dGamma3)  # d_d Gamma3^a_bc
    term3 = jnp.einsum("ace,ebd->abcd", Gamma3, Gamma3)
    term4 = jnp.einsum("ade,ebc->abcd", Gamma3, Gamma3)
    Riemann3 = term1 - term2 + term3 - term4   # R^a_bcd (3D)
    Ricci3 = jnp.einsum("abad->bd", Riemann3)  # contract a with c
    return Ricci3


# ---------------------------------------------------------------------------
# Extrinsic curvature, its spatial gradient, and its exact time-derivative
# (all via autodiff through the closed-form metric -- no finite differences,
# per the project's standing "autodiff over FD near sharp features" rule).
# ---------------------------------------------------------------------------

def _K_of_coords(coords, params: WarpBubbleParams):
    """K_ij(t,x,y,z) = -Gamma^0_ij, the same closed form already validated
    (against the independent shift-vector formula) in adm.py."""
    _, _, Gamma = christoffel_at_point(coords, params, engine="autodiff")
    return -Gamma[0, 1:4, 1:4]


def extrinsic_curvature_and_derivatives(t, xyz, params: WarpBubbleParams):
    """Returns K_ij, its exact spatial gradient dK[i,j,k] = d_k K_ij, and
    its exact time derivative dK_dt[i,j] = d(K_ij)/dt, all via autodiff."""
    def K_of_xyz(x):
        coords = jnp.concatenate([jnp.array([t], dtype=x.dtype), x])
        return _K_of_coords(coords, params)

    def K_of_t(tt):
        coords = jnp.concatenate([jnp.array([tt], dtype=xyz.dtype), xyz])
        return _K_of_coords(coords, params)

    K = K_of_xyz(xyz)
    dK_spatial = jax.jacfwd(K_of_xyz)(xyz)  # dK_spatial[i,j,k] = d_k K_ij
    dK_dt = jax.jacfwd(K_of_t)(t)           # dK_dt[i,j] = d(K_ij)/dt
    return K, dK_spatial, dK_dt


def shift_and_gradient(t, xyz, params: WarpBubbleParams):
    """beta^i(x,y,z) at fixed t, plus its spatial gradient
    dbeta[k,j] = d(beta^k)/dx^j."""
    def beta_of_xyz(x):
        coords = jnp.concatenate([jnp.array([t], dtype=x.dtype), x])
        g = metric_tensor(coords, params)
        _, _, beta_up, _ = adm_decompose(g)
        return beta_up

    beta_up = beta_of_xyz(xyz)
    dbeta = jax.jacfwd(beta_of_xyz)(xyz)  # dbeta[k,j] = d(beta^k)/dx^j
    return beta_up, dbeta


def lapse_hessian(t, xyz, params: WarpBubbleParams, Gamma3):
    """Covariant Hessian D_i D_j alpha = d_i d_j alpha - (3)Gamma^k_ij d_k alpha."""
    def alpha_of_xyz(x):
        coords = jnp.concatenate([jnp.array([t], dtype=x.dtype), x])
        g = metric_tensor(coords, params)
        alpha, _, _, _ = adm_decompose(g)
        return alpha

    grad_alpha = jax.grad(alpha_of_xyz)(xyz)
    hess_alpha = jax.hessian(alpha_of_xyz)(xyz)
    D2alpha = hess_alpha - jnp.einsum("kij,k->ij", Gamma3, grad_alpha)
    return D2alpha


def matter_projection(coords, params: WarpBubbleParams):
    """rho, S_ij, S = gamma^ij S_ij for the Eulerian observer, built from
    the general projector gamma^a_b = delta^a_b + n^a n_b (not a
    metric-specific shortcut)."""
    out = full_curvature_at_point(coords, params, engine="autodiff")
    T = stress_energy_tensor(out["Einstein"])
    g = out["g"]
    n = normalize_eulerian_observer(g, check=False)
    rho = energy_density(T, n)
    n_lower = g @ n  # n_a = g_ab n^b
    proj = jnp.eye(4, dtype=g.dtype) + jnp.outer(n, n_lower)  # gamma^a_b
    S_full = jnp.einsum("ca,db,cd->ab", proj, proj, T)  # S_ab = gamma^c_a gamma^d_b T_cd
    S_ij = S_full[1:4, 1:4]
    gamma_ij = g[1:4, 1:4]
    gamma_inv = jnp.linalg.inv(gamma_ij)
    S = jnp.einsum("ij,ij->", gamma_inv, S_ij)
    return rho, S_ij, S


# ---------------------------------------------------------------------------
# The full ADM evolution equation and its cross-check.
# ---------------------------------------------------------------------------

def kij_evolution_rhs(coords, params: WarpBubbleParams):
    """RHS of d/dt K_ij (the full nonlinear ADM evolution equation),
    evaluated at spacetime point `coords` = (t,x,y,z)."""
    t = coords[0]
    xyz = coords[1:4]

    g = metric_tensor(coords, params)
    alpha, _, _, _ = adm_decompose(g)
    gamma, gamma_inv, Gamma3 = spatial_christoffel(t, xyz, params)
    Ricci3 = spatial_ricci(t, xyz, params)

    K, dK_spatial, _ = extrinsic_curvature_and_derivatives(t, xyz, params)
    beta_up, dbeta = shift_and_gradient(t, xyz, params)
    D2alpha = lapse_hessian(t, xyz, params, Gamma3)

    K_trace = jnp.trace(K)
    KK = K @ K  # K_ik gamma^kl K_lj, trivial index raise here (gamma finite-diff'd generally below)
    KK = jnp.einsum("ik,kl,lj->ij", K, gamma_inv, K)

    advection = jnp.einsum("k,ijk->ij", beta_up, dK_spatial)          # beta^k d_k K_ij
    shift_coupling = (
        jnp.einsum("ik,kj->ij", K, dbeta)                             # K_ik d_j beta^k
        + jnp.einsum("kj,ki->ij", K, dbeta)                            # K_kj d_i beta^k
    )

    rho, S_ij, S = matter_projection(coords, params)
    matter_term = S_ij - 0.5 * gamma * (S - rho)

    rhs = (
        advection + shift_coupling
        - D2alpha
        + alpha * (Ricci3 + K_trace * K - 2.0 * KK)
        - 8.0 * jnp.pi * alpha * matter_term
    )
    return rhs


def evolution_equation_residual(coords, params: WarpBubbleParams):
    """The core self-consistency check: RHS of the ADM evolution equation
    minus the exact autodiff time-derivative of K_ij. Should be ~0 to
    machine precision at every point, confirming the metric genuinely
    satisfies the *dynamical* Einstein equations under 3+1 evolution (not
    just the constraints)."""
    t = coords[0]
    xyz = coords[1:4]
    _, _, dK_dt = extrinsic_curvature_and_derivatives(t, xyz, params)
    rhs = kij_evolution_rhs(coords, params)
    return dK_dt - rhs
