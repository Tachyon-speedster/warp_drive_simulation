"""
adm.py — 3+1 (ADM) View of the Spacetime (project doc, section 14)

MATH
----
Any 4-metric can be decomposed into "space + time" (ADM form):

    ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)

where:
    alpha    = lapse function       (rate coordinate time elapses relative
                                      to proper time of a normal observer)
    beta^i   = shift vector         (how spatial coordinates are "dragged"
                                      between successive time slices)
    gamma_ij = spatial 3-metric     (induced metric on each t=const slice)

Reading off alpha, beta^i, gamma_ij from a general g_{ab}:

    gamma_ij = g_ij
    beta_i   = g_{0i}          (beta^i = gamma^{ij} beta_j)
    alpha^2  = beta_i beta^i - g_{00}

For the Alcubierre metric specifically (metric.py), by construction:

    alpha = 1                         (unit lapse — "Alcubierre gauge")
    beta^x = -v_s(t) f(r_s),  beta^y = beta^z = 0
    gamma_ij = delta_ij               (flat spatial slices — this is the
                                        special/defining property of the
                                        Alcubierre construction: spatial
                                        curvature is exactly zero, ALL of
                                        the curvature lives in how the
                                        shift vector varies across the
                                        slice)

This decomposition is what numerical-relativity evolution codes (e.g. BSSN,
ADM formalism itself) integrate forward in time. This module doesn't evolve
anything yet (that is "Advanced" milestone 21-23 in the project doc); it
only exposes the split for diagnostics/visualization and to prepare for it.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

from .metric import metric_tensor
from .christoffel import christoffel_at_point


def adm_decompose(g):
    """Given g_{ab} (4x4), return (alpha, beta_i, beta^i, gamma_ij)."""
    gamma_ij = g[1:4, 1:4]
    beta_i = g[0, 1:4]
    gamma_inv = jnp.linalg.inv(gamma_ij)
    beta_up = gamma_inv @ beta_i
    beta_sq = jnp.dot(beta_i, beta_up)
    alpha_sq = beta_sq - g[0, 0]
    alpha = jnp.sqrt(alpha_sq)
    return alpha, beta_i, beta_up, gamma_ij


def extrinsic_curvature(coords, params, engine="autodiff"):
    """Extrinsic curvature K_ij of the t=const slice (project doc section
    21-22: "study extrinsic curvature").

    DERIVATION (kept explicit here because sign conventions for K_ij are
    notoriously inconsistent across GR textbooks -- this one is derived
    from first principles rather than copied from a particular convention):

    The unit normal covector to a t=const slice is n_a = -alpha (dt)_a,
    i.e. n_a = (-alpha, 0, 0, 0). For the Alcubierre metric, alpha = 1
    EVERYWHERE (metric.py), so n_a = (-1,0,0,0) is a CONSTANT covector
    field (no position dependence at all). That makes the covariant
    derivative trivial to evaluate:

        nabla_a n_b = d_a n_b - Gamma^c_{ab} n_c
                    = 0 - Gamma^c_{ab} * (-1) * delta_{c,0}   [only n_0=-1 is nonzero]
                    = Gamma^0_{ab}

    Extrinsic curvature is (with the sign convention K_ab = -h_a^c h_b^d
    nabla_c n_d, projected onto the slice via h_ab = g_ab + n_a n_b): for
    purely SPATIAL a=i, b=j, the projector h_i^a is just delta_i^a (since
    n_i=0 for spatial i -- there is no time-component mixing to project
    away), so this simplifies to the clean closed form:

        K_ij = -nabla_i n_j = -Gamma^0_{ij}

    which needs nothing beyond the Christoffel symbols we already compute.
    This was cross-checked numerically against the textbook ADM formula
    K_ij = (1/2)(d_i beta_j + d_j beta_i) (valid here because alpha=1 and
    gamma_ij=delta_ij is time-independent) -- the two agree to autodiff
    (machine) precision at every test point.
    """
    _, _, Gamma = christoffel_at_point(coords, params, engine=engine)
    K = -Gamma[0, 1:4, 1:4]
    return K


def extrinsic_curvature_from_shift(coords, params):
    """Cross-check: K_ij = (1/2)(d_i beta_j + d_j beta_i), valid because
    alpha=1 and gamma_ij=delta_ij (time-independent) for this metric
    family. Used only in validation.py to confirm the two independent
    derivations of K_ij agree."""
    def beta_of_coords(c):
        g = metric_tensor(c, params)
        _, beta_i, _, _ = adm_decompose(g)
        return beta_i  # beta_i = g_{0i}; gamma_ij=delta_ij => beta_j==beta^j

    dbeta = jax.jacfwd(beta_of_coords)(coords)  # dbeta[i,k] = d(beta_i)/dx^k
    dbeta_spatial = dbeta[:, 1:4]  # (i,j) = d(beta_i)/dx^j
    K = 0.5 * (dbeta_spatial + dbeta_spatial.T)
    return K


def hamiltonian_constraint_residual(coords, params, engine="autodiff"):
    """ADM Hamiltonian constraint (project doc section 23):

        (3)R + K^2 - K_ij K^ij = 16*pi*rho_ADM

    where (3)R is the Ricci scalar of the SPATIAL metric gamma_ij (which
    is identically 0 here, since gamma_ij=delta_ij is flat), K = trace(K_ij),
    and rho_ADM = T_ab n^a n^b is the energy density measured by the
    Eulerian (ADM-normal) observer. Because our T_ab was itself derived
    purely from the geometry via Einstein's equations (stress_energy.py),
    this residual is a genuine, independent numerical-relativity
    self-consistency check on the whole pipeline: if it's not ~0 to
    numerical precision, something upstream (Christoffel/Riemann/Ricci/
    Einstein) has a bug. Returns the residual (should be ~0)."""
    K = extrinsic_curvature(coords, params, engine=engine)
    K_trace = jnp.trace(K)
    K_sq = jnp.sum(K * K)  # K_ij K^ij, trivial index raising (gamma_ij=delta_ij)
    g = metric_tensor(coords, params)
    from .curvature import full_curvature_at_point
    from .stress_energy import stress_energy_tensor
    from .observer import normalize_eulerian_observer, energy_density
    out = full_curvature_at_point(coords, params, engine=engine)
    T = stress_energy_tensor(out["Einstein"])
    n = normalize_eulerian_observer(g, check=False)
    rho_adm = energy_density(T, n)
    lhs = K_trace ** 2 - K_sq  # (3)R = 0
    rhs = 16.0 * jnp.pi * rho_adm
    return lhs - rhs


def momentum_constraint_residual(coords, params, engine="autodiff"):
    """ADM momentum constraint (project doc section 23):

        D_j(K^ij - gamma^ij K) = 8*pi*j^i

    where D_j is the spatial covariant derivative (= partial derivative
    here, since gamma_ij=delta_ij is flat) and j_i = -T_ib n^b is the
    momentum density measured by the Eulerian observer. Returns a length-3
    residual vector (should be ~0 componentwise)."""
    def K_of_coords(c):
        _, _, Gamma = christoffel_at_point(c, params, engine=engine)
        return -Gamma[0, 1:4, 1:4]

    K = K_of_coords(coords)
    dK = jax.jacfwd(K_of_coords)(coords)      # dK[i,j,k] = d(K_ij)/dx^k, k in 0..3
    dK_spatial = dK[:, :, 1:4]                # keep spatial derivative index only

    K_trace = jnp.trace(K)
    dK_trace = jnp.einsum("iik->k", dK_spatial)          # d(K_trace)/dx^k, spatial k
    div_term = jnp.einsum("ijj->i", dK_spatial)          # sum_j d(K_ij)/dx^j
    lhs = div_term - dK_trace                             # D_j(K^ij - gamma^ij K)

    g = metric_tensor(coords, params)
    from .curvature import full_curvature_at_point
    from .stress_energy import stress_energy_tensor
    from .observer import normalize_eulerian_observer
    out = full_curvature_at_point(coords, params, engine=engine)
    T = stress_energy_tensor(out["Einstein"])
    n = normalize_eulerian_observer(g, check=False)
    j_i = -jnp.einsum("ib,b->i", T[1:4, :], n)
    rhs = 8.0 * jnp.pi * j_i
    return lhs - rhs
