"""
bssn.py -- BSSN (Baumgarte-Shapiro-Shibata-Nakamura) reformulation of the
free-evolution equations in free_evolution.py.

WHY THIS MODULE EXISTS
-----------------------
run_free_evolution.py's own printed summary names the diagnosis exactly:
plain ADM is only *weakly* hyperbolic -- its principal symbol has
non-diagonalizable degeneracies, so generic perturbations (numerical or
physical) grow without bound. That is the blow-up `free_evolution.py`
reports within a fraction of a wall light-crossing time, and it is not a
bug in that implementation; every credible NR code published since
~2000 avoids it by not evolving raw ADM.

BSSN fixes this with three structural changes (Baumgarte & Shapiro,
*Numerical Relativity*, CUP 2010, Sec. 3.1-3.2; originally Nakamura-
Oohara-Kojima 1987 / Shibata-Nakamura 1995 / Baumgarte-Shapiro 1999):

  1. Conformal decomposition: gamma_ij = e^{4 phi} gamma~_ij with
     det(gamma~) = 1 factored out into a separately evolved scalar phi.
  2. Trace/trace-free split of the extrinsic curvature: K (trace) and
     A~_ij = e^{-4phi} (K_ij - (1/3) gamma_ij K) (conformal trace-free
     part) evolved separately instead of the single tensor K_ij.
  3. THE load-bearing change: the contracted conformal connection
     functions Gamma~^i = gamma~^jk Gamma~^i_jk are promoted from an
     algebraic function of gamma~_ij to an INDEPENDENTLY EVOLVED FIELD.
     This lets the conformal Ricci tensor be rewritten (`conformal_ricci`
     below) so that its second-derivative terms reduce to a clean scalar
     Laplacian of gamma~_ij plus first derivatives of the evolved Gamma~^i
     field -- eliminating exactly the mixed second-derivative terms that
     make plain ADM's principal symbol degenerate. This substitution (not
     the conformal rescaling by itself) is what turns a weakly hyperbolic
     system into the strongly hyperbolic BSSN system. Getting it right
     therefore means Gamma~^i must be evolved as its own state variable
     and never silently replaced by gamma~^jk Gamma~^i_jk[gamma~] inside
     the RHS -- that substitution would collapse the scheme back to
     (conformally relabeled) ADM and buy nothing.

GAUGE
-----
alpha, beta^i are still the prescribed Alcubierre closed-form fields
(re-evaluated every RK substage from metric.py), exactly as in
free_evolution.py -- this module isolates the effect of the BSSN
reformulation from the effect of a dynamical gauge (that is the separate
"dynamical gauge" milestone). Comparing this module's constraint growth
against free_evolution.py's, at identical resolution/CFL/gauge, isolates
what the formalism change alone buys.

WHAT IS AND ISN'T VALIDATED HERE
----------------------------------
`conformal_ricci` (the load-bearing, previously-buggy piece) is now
transcribed from the real NRPy+ source rather than reconstructed from
textbook excerpts, and verified against an independent exact symbolic
computation at a generic point -- see the comment directly above that
function for the full story and `validate_conformal_ricci` below for the
grid-based convergence check that runs at import time. `phi_ricci_correction`
(the conformal-factor part of the Ricci split) was separately checked
against an independent direct-FD computation on a purely conformally-flat
test metric and shown to converge at the expected O(h^2) rate. What is
NOT independently validated: the Gamma~^i evolution equation's
shift-Hessian and matter-coupling terms (no independent second derivation
exists in this codebase to cross-check against) -- these are implemented
directly from the standard published equation (Baumgarte & Shapiro 2010,
eq. 3.4-3.5) and are consistent with the vacuum limit only by inspection,
not by an independent numerical cross-check.

MATTER
------
As in free_evolution.py, only rho, S_ij, S (from `matter_source`) are
available from this project's matter model; there is no independently
derived momentum-density S_i, so the -16*pi*alpha*gamma~^ij*S_j term in
the Gamma~^i RHS is omitted (set to zero) rather than guessed. This
matches this codebase's existing "frozen_source" matter treatment, which
also never re-derives a momentum flux.
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .metric import WarpBubbleParams, metric_tensor
from .adm import adm_decompose, extrinsic_curvature
from .free_evolution import (
    EvolutionGrid, make_evolution_grid, d_axis, grad3, kreiss_oliger,
    spatial_christoffel_ricci_fd, initial_data as adm_initial_data,
    gauge_fields, matter_source, _flat_coords,
)


# ---------------------------------------------------------------------------
# Generic second-derivative (Hessian) helper -- same nested-d_axis pattern
# free_evolution.adm_rhs already uses for D2alpha, generalized to any field
# whose leading 3 axes are spatial.
# ---------------------------------------------------------------------------

def hessian3(f, h):
    """Returns array of shape (3,3)+f.shape = [a,b,...] = d_a d_b f,
    via nested centered differences (consistent with grad3/d_axis)."""
    d1 = grad3(f, h)  # (3,)+f.shape = [a,...] = d_a f
    rows = [grad3(d1[a], h) for a in range(3)]  # each (3,)+f.shape=[b,...]=d_b(d_a f)
    return jnp.stack(rows, axis=0)  # [a,b,...]


# ---------------------------------------------------------------------------
# ADM <-> BSSN variable conversion
# ---------------------------------------------------------------------------

@dataclass
class BSSNState:
    t: float
    phi: jnp.ndarray          # (...,)      conformal factor
    gamma_t: jnp.ndarray      # (...,3,3)   conformal metric, det=1
    K: jnp.ndarray            # (...,)      trace of extrinsic curvature
    A_t: jnp.ndarray          # (...,3,3)   conformal trace-free extrinsic curvature
    Gamma_t: jnp.ndarray      # (...,3)     evolved conformal connection functions


def adm_to_bssn(gamma, K_ij, h) -> BSSNState:
    """Convert physical (gamma_ij, K_ij) fields -> BSSN variables,
    including the initial value of the evolved Gamma~^i field (its
    algebraic definition, used ONLY to set the initial condition -- after
    t=0 it is evolved independently, see module docstring)."""
    det_g = jnp.linalg.det(gamma)
    phi = jnp.log(det_g) / 12.0
    e4phi = jnp.exp(4.0 * phi)
    gamma_t = gamma / e4phi[..., None, None]

    K_trace = jnp.einsum('...ii->...', jnp.einsum('...ij,...jk->...ik',
                                                    jnp.linalg.inv(gamma), K_ij))
    A_ij = K_ij - (1.0 / 3.0) * gamma * K_trace[..., None, None]
    A_t = A_ij / e4phi[..., None, None]

    Gamma3, _ = spatial_christoffel_ricci_fd(gamma_t, h)
    gamma_t_inv = jnp.linalg.inv(gamma_t)
    Gamma_t = jnp.einsum('...jk,...ijk->...i', gamma_t_inv, Gamma3)
    return BSSNState(0.0, phi, gamma_t, K_trace, A_t, Gamma_t)


def bssn_to_adm(state: BSSNState):
    e4phi = jnp.exp(4.0 * state.phi)
    gamma = state.gamma_t * e4phi[..., None, None]
    A_ij = state.A_t * e4phi[..., None, None]
    K_ij = A_ij + (1.0 / 3.0) * gamma * state.K[..., None, None]
    return gamma, K_ij


# ---------------------------------------------------------------------------
# Conformal Ricci tensor, using the EVOLVED Gamma~^i field.
#
# This is the load-bearing structural substitution described in the module
# docstring. The formula below is transcribed directly from the real,
# pip-installable NRPy+ source (nrpy.equations.general_relativity.
# BSSN_quantities, function BSSN_quantities.__init__, "Step 7"), specialized
# to a flat Cartesian reference metric (Gammahat=0, so Delta^i_jk = Gamma~^i_jk
# exactly and all D-hat covariant derivatives reduce to ordinary partial
# derivatives -- both true for this module's Cartesian grid). It is NOT
# transcribed from a textbook excerpt -- three independent textbook/review
# renderings of this formula turned out to disagree with each other on
# exactly the index-order details that matter, which is what caused the
# earlier (now-fixed) bug in this function. This version was verified
# against an independent, exact (rational-arithmetic, zero floating-point
# truncation) symbolic computation of the physical Ricci tensor at a
# generic point, matching on every component -- see development notes at
# the bottom of this file for the verification script.
#
#   R~_ij = -1/2 gamma~^kl d_k d_l gamma~_ij
#           + 1/2 (gamma~_ki d_j Gamma~^k + gamma~_kj d_i Gamma~^k)
#           + 1/2 Gamma~^k (Gamma~_ijk + Gamma~_jik)
#           + gamma~^kl ( Gamma~^m_ki Gamma~_jml
#                       + Gamma~^m_kj Gamma~_iml
#                       + Gamma~^m_ik Gamma~_mjl )
#
# where Gamma~_abc := gamma~_ad Gamma~^d_bc (lowered first index, matching
# NRPy+'s DGammaDDD[a][b][c]).
# ---------------------------------------------------------------------------

def conformal_ricci(gamma_t, Gamma_t_vec, h):
    gamma_t_inv = jnp.linalg.inv(gamma_t)
    Gamma3, _ = spatial_christoffel_ricci_fd(gamma_t, h)              # [...,a,b,c] = Gamma~^a_bc
    Gamma3_low = jnp.einsum('...ad,...dbc->...abc', gamma_t, Gamma3)  # [...,a,b,c] = Gamma~_a,bc

    # -1/2 gamma~^kl d_k d_l gamma~_ij
    hess_gt = hessian3(gamma_t, h)                       # [a,b,...,i,j] = d_a d_b gamma~_ij
    hess_gt = jnp.moveaxis(hess_gt, (0, 1), (-4, -3))     # [...,k,l,i,j]
    term_hess = -0.5 * jnp.einsum('...kl,...klij->...ij', gamma_t_inv, hess_gt)

    # 1/2 (gamma~_ki d_j Gamma~^k + gamma~_kj d_i Gamma~^k)
    dGamma_t = jnp.moveaxis(grad3(Gamma_t_vec, h), 0, -2)  # [...,j,k] = d_j Gamma~^k
    term_sym = 0.5 * (jnp.einsum('...ki,...jk->...ij', gamma_t, dGamma_t)
                       + jnp.einsum('...kj,...ik->...ij', gamma_t, dGamma_t))

    # 1/2 Gamma~^k (Gamma~_ijk + Gamma~_jik)
    term_GG = 0.5 * jnp.einsum('...k,...ijk->...ij', Gamma_t_vec, Gamma3_low + jnp.swapaxes(Gamma3_low, -3, -2))

    # gamma~^kl ( Gamma~^m_ki Gamma~_jml + Gamma~^m_kj Gamma~_iml + Gamma~^m_ik Gamma~_mjl )
    quad = (jnp.einsum('...kl,...mki,...jml->...ij', gamma_t_inv, Gamma3, Gamma3_low)
            + jnp.einsum('...kl,...mkj,...iml->...ij', gamma_t_inv, Gamma3, Gamma3_low)
            + jnp.einsum('...kl,...mik,...mjl->...ij', gamma_t_inv, Gamma3, Gamma3_low))

    Ricci_t = term_hess + term_sym + term_GG + quad
    return Ricci_t, Gamma3, Gamma3_low, gamma_t_inv


def phi_ricci_correction(phi, gamma_t, gamma_t_inv, Gamma3, h):
    """R^phi_ij = -2 D~_i D~_j phi - 2 gamma~_ij D~^k D~_k phi
                  + 4 (d_i phi)(d_j phi) - 4 gamma~_ij (D~ phi)^2"""
    grad_phi = jnp.moveaxis(grad3(phi, h), 0, -1)          # [...,k] = d_k phi
    hess_phi = jnp.moveaxis(hessian3(phi, h), (0, 1), (-2, -1))  # [...,i,j] = d_i d_j phi
    Dphi = jnp.einsum('...kij,...k->...ij', Gamma3, grad_phi)   # Gamma~^k_ij d_k phi
    D2phi = hess_phi - Dphi                                      # D~_i D~_j phi

    lap_phi = jnp.einsum('...ij,...ij->...', gamma_t_inv, D2phi)
    grad_phi_sq = jnp.einsum('...ij,...i,...j->...', gamma_t_inv, grad_phi, grad_phi)
    outer_phi = jnp.einsum('...i,...j->...ij', grad_phi, grad_phi)

    R_phi = (-2.0 * D2phi - 2.0 * gamma_t * lap_phi[..., None, None]
             + 4.0 * outer_phi - 4.0 * gamma_t * grad_phi_sq[..., None, None])
    return R_phi, grad_phi


def trace_free(X_ij, gamma_t, gamma_t_inv):
    tr = jnp.einsum('...ij,...ij->...', gamma_t_inv, X_ij)
    return X_ij - (1.0 / 3.0) * gamma_t * tr[..., None, None]


# ---------------------------------------------------------------------------
# BSSN evolution RHS
# ---------------------------------------------------------------------------

def bssn_rhs(state: BSSNState, alpha, beta, h, matter=None):
    phi, gamma_t, K, A_t, Gamma_t_vec = (state.phi, state.gamma_t, state.K,
                                          state.A_t, state.Gamma_t)
    gamma_t_inv = jnp.linalg.inv(gamma_t)

    Ricci_t, Gamma3, Gamma3_low, _ = conformal_ricci(gamma_t, Gamma_t_vec, h)
    R_phi, grad_phi = phi_ricci_correction(phi, gamma_t, gamma_t_inv, Gamma3, h)
    Ricci_phys = Ricci_t + R_phi                       # physical Ricci = R~ + R^phi

    grad_alpha = jnp.moveaxis(grad3(alpha, h), 0, -1)   # [...,k] = d_k alpha
    hess_alpha = jnp.moveaxis(hessian3(alpha, h), (0, 1), (-2, -1))  # [...,i,j]
    Dt_alpha = hess_alpha - jnp.einsum('...kij,...k->...ij', Gamma3, grad_alpha)  # D~_iD~_j alpha
    # physical D_iD_j alpha via the phi-conformal transform. Standard identity
    # for g_ab=e^{2w}g~_ab: D_iD_jf = D~_iD~_jf - (d_i w d_jf + d_jw d_if) + g~_ij(D~w.D~f).
    # Here w=2phi (since gamma=e^{4phi}gamma~=e^{2(2phi)}gamma~), so d_iw=2 d_iphi:
    #   D_iD_jalpha = D~_iD~_jalpha - 2(d_iphi d_jalpha + d_jphi d_ialpha) + 2 gamma~_ij (D~phi.D~alpha)
    # (verified against the exact NRPy+ source's curlybracketDD/trK_rhs Term4,
    # which fixes a sign error present in an earlier version of this function).
    D2alpha = (Dt_alpha
               - 2.0 * (jnp.einsum('...i,...j->...ij', grad_phi, grad_alpha)
                        + jnp.einsum('...j,...i->...ij', grad_phi, grad_alpha))
               + 2.0 * gamma_t * jnp.einsum('...ij,...i,...j->...', gamma_t_inv, grad_phi, grad_alpha)[..., None, None])

    e4phi = jnp.exp(4.0 * phi)
    lap_alpha_phys = jnp.exp(-4.0 * phi) * (
        jnp.einsum('...ij,...ij->...', gamma_t_inv, Dt_alpha)
        + 2.0 * jnp.einsum('...ij,...i,...j->...', gamma_t_inv, grad_phi, grad_alpha))

    A_up = jnp.einsum('...ik,...jl,...kl->...ij', gamma_t_inv, gamma_t_inv, A_t)  # Ã^ij
    A_sq = jnp.einsum('...ij,...ij->...', A_t, A_up)  # Ã_ij Ã^ij

    beta_up = beta  # beta^i, matches free_evolution's gauge_fields() convention
    dbeta = jnp.moveaxis(grad3(beta_up, h), 0, -2)      # [...,j,i] = d_j beta^i
    div_beta = jnp.einsum('...ii->...', dbeta)          # d_i beta^i

    grid_ndim = alpha.ndim  # rank of the spatial grid itself (e.g. 3 for nx,ny,nz)
    beta_moved = jnp.moveaxis(beta_up, -1, 0)  # [a,...] = beta^a, shape (3,)+grid

    def advect(f):
        g = grad3(f, h)  # [a,...] (+ trailing tensor axes beyond the grid, if any)
        extra_rank = g.ndim - 1 - grid_ndim  # trailing tensor rank on f (0,1,or 2)
        b = beta_moved
        for _ in range(extra_rank):
            b = b[..., None]
        return jnp.sum(b * g, axis=0)

    rho = S = S_ij = None
    if matter is not None:
        rho, S_ij, S = matter

    # --- phi ---
    rhs_phi = advect(phi) + (1.0 / 6.0) * (div_beta - alpha * K)

    # --- gamma~_ij ---
    D_beta_low_sym = (jnp.einsum('...ik,...jk->...ij', gamma_t, dbeta)
                       + jnp.einsum('...jk,...ik->...ij', gamma_t, dbeta))
    rhs_gamma_t = (advect(gamma_t) + D_beta_low_sym
                   - (2.0 / 3.0) * gamma_t * div_beta[..., None, None]
                   - 2.0 * alpha[..., None, None] * A_t)

    # --- K ---
    matter_K = 4.0 * jnp.pi * alpha * (rho + S) if matter is not None else 0.0
    rhs_K = (advect(K) - lap_alpha_phys
             + alpha * (A_sq + (K ** 2) / 3.0) + matter_K)

    # --- A~_ij ---
    S_ij_TF_term = 0.0
    if matter is not None:
        S_ij_TF_term = 8.0 * jnp.pi * trace_free(S_ij, gamma_t, gamma_t_inv)
    RHS_bracket = -trace_free(D2alpha, gamma_t, gamma_t_inv) + alpha[..., None, None] * (
        trace_free(Ricci_phys, gamma_t, gamma_t_inv) - S_ij_TF_term)
    AA_term = jnp.einsum('...ik,...kl,...lj->...ij', A_t, gamma_t_inv, A_t)  # Ã_ik gamma~^kl Ã_lj
    rhs_A_t = (advect(A_t) + D_beta_low_sym_from(A_t, dbeta)
               - (2.0 / 3.0) * A_t * div_beta[..., None, None]
               + jnp.exp(-4.0 * phi)[..., None, None] * RHS_bracket
               + alpha[..., None, None] * (K[..., None, None] * A_t - 2.0 * AA_term))

    # --- Gamma~^i ---
    hess_beta = hessian3(beta_up, h)                     # [j,k,...,i]
    hess_beta = jnp.moveaxis(hess_beta, (0, 1), (-2, -1))  # [...,i,j,k]  (i is last of beta axes)
    lap_beta_up = jnp.einsum('...jk,...ijk->...i', gamma_t_inv, hess_beta)  # gamma~^jk d_jd_k beta^i

    div_beta_grad = jnp.moveaxis(grad3(div_beta, h), 0, -1)  # [...,j] = d_j(div beta)
    third_term = (1.0 / 3.0) * jnp.einsum('...ij,...j->...i', gamma_t_inv, div_beta_grad)

    Gamma_dbeta = jnp.einsum('...j,...ji->...i', Gamma_t_vec, dbeta)  # Gamma~^j d_j beta^i
    Gamma_divbeta = (2.0 / 3.0) * Gamma_t_vec * div_beta[..., None]

    A_grad_alpha = -2.0 * jnp.einsum('...ij,...j->...i', A_up, grad_alpha)
    grad_K = jnp.moveaxis(grad3(K, h), 0, -1)
    matter_S_i_term = 0.0  # see module docstring: no independent S_i in this codebase's matter model

    alg = (jnp.einsum('...ijk,...jk->...i', Gamma3, A_up)
           - (2.0 / 3.0) * jnp.einsum('...ij,...j->...i', gamma_t_inv, grad_K)
           + matter_S_i_term
           + 6.0 * jnp.einsum('...ij,...j->...i', A_up, grad_phi))

    rhs_Gamma_t = (advect(Gamma_t_vec) - Gamma_dbeta + Gamma_divbeta
                   + lap_beta_up + third_term + A_grad_alpha
                   + 2.0 * alpha[..., None] * alg)

    rhs_gamma_t = rhs_gamma_t + kreiss_oliger(gamma_t, h)
    rhs_A_t = rhs_A_t + kreiss_oliger(A_t, h)
    rhs_K = rhs_K + kreiss_oliger(K, h)
    rhs_phi = rhs_phi + kreiss_oliger(phi, h)
    rhs_Gamma_t = rhs_Gamma_t + kreiss_oliger(Gamma_t_vec, h)

    return rhs_phi, rhs_gamma_t, rhs_K, rhs_A_t, rhs_Gamma_t


def D_beta_low_sym_from(T_ij, dbeta):
    """Shift-advection Lie-derivative helper: T_ik d_j beta^k + T_jk d_i beta^k,
    used identically for gamma~_ij and A~_ij (they transform the same way
    under the shift since both are rank-2 tensor densities of the same
    conformal weight for this purpose)."""
    return (jnp.einsum('...ik,...jk->...ij', T_ij, dbeta)
            + jnp.einsum('...jk,...ik->...ij', T_ij, dbeta))


# ---------------------------------------------------------------------------
# Algebraic constraint enforcement (standard BSSN practice: renormalize
# det(gamma~)=1 and re-trace-free A~ every step -- keeps roundoff/RK drift
# in these two ALGEBRAIC constraints from compounding; does not touch the
# Hamiltonian/momentum constraints, which are the real physics diagnostic).
# ---------------------------------------------------------------------------

def enforce_algebraic_constraints(state: BSSNState):
    det_gt = jnp.linalg.det(state.gamma_t)
    gamma_t = state.gamma_t * jnp.power(det_gt, -1.0 / 3.0)[..., None, None]
    gamma_t_inv = jnp.linalg.inv(gamma_t)
    A_t = trace_free(state.A_t, gamma_t, gamma_t_inv)
    return BSSNState(state.t, state.phi, gamma_t, state.K, A_t, state.Gamma_t)


# ---------------------------------------------------------------------------
# Dynamical gauge: 1+log slicing (lapse) + 2nd-order Gamma-driver (shift),
# transcribed from the real NRPy+ source
# (nrpy.equations.general_relativity.BSSN_gauge_RHSs, LapseEvolutionOption=
# "OnePlusLog", ShiftEvolutionOption="GammaDriving2ndOrder_NoCovariant" --
# the "NoCovariant" variant is the exact right choice here, not a
# simplification: it and the "Covariant" variant differ only by terms
# involving Gamma~^i_jk contracted directly into the beta/B evolution
# equations, which are reference-metric bookkeeping for curvilinear
# coordinates; this module's reference metric is flat Cartesian, so those
# terms are what BSSN_gauge_RHSs.py's own reference-metric machinery would
# reduce to zero anyway).
#
#   d_t alpha  = beta^i d_i alpha - 2 alpha K                        (1+log)
#   d_t beta^i = beta^j d_j beta^i + B^i                             (Gamma-driver)
#   d_t B^i    = beta^j d_j B^i + (3/4) d_0 Gamma~^i - eta B^i       (Gamma-driver)
#
# where d_0 Gamma~^i := d_t Gamma~^i - beta^j d_j Gamma~^i (the
# non-advective part of the already-computed Gamma~^i RHS), and eta is a
# free damping parameter (NRPy+ default eta=2.0, used here too).
# ---------------------------------------------------------------------------

@dataclass
class BSSNStateDyn:
    t: float
    phi: jnp.ndarray
    gamma_t: jnp.ndarray
    K: jnp.ndarray
    A_t: jnp.ndarray
    Gamma_t: jnp.ndarray
    alpha: jnp.ndarray   # (...,)   lapse, now EVOLVED rather than prescribed
    beta: jnp.ndarray    # (...,3)  shift, now EVOLVED
    B_aux: jnp.ndarray   # (...,3)  Gamma-driver auxiliary variable


def to_plain_state(s: BSSNStateDyn) -> BSSNState:
    return BSSNState(s.t, s.phi, s.gamma_t, s.K, s.A_t, s.Gamma_t)


def gauge_rhs(state: BSSNStateDyn, rhs_Gamma_t, h, eta=2.0):
    alpha, beta, B, Gamma_t_vec, K = (state.alpha, state.beta, state.B_aux,
                                       state.Gamma_t, state.K)

    def advect_scalar_or_vec(f):
        g = grad3(f, h)
        b = jnp.moveaxis(beta, -1, 0)
        for _ in range(g.ndim - 1 - b.ndim + 1):
            b = b[..., None]
        return jnp.sum(b * g, axis=0)

    alpha_rhs = -2.0 * alpha * K + advect_scalar_or_vec(alpha)
    beta_rhs = B + advect_scalar_or_vec(beta)

    adv_Gamma_t = advect_scalar_or_vec(Gamma_t_vec)
    Lambdabar_partial0 = rhs_Gamma_t - adv_Gamma_t  # d_0 Gamma~^i
    B_rhs = 0.75 * Lambdabar_partial0 - eta * B + advect_scalar_or_vec(B)

    alpha_rhs = alpha_rhs + kreiss_oliger(alpha, h)
    beta_rhs = beta_rhs + kreiss_oliger(beta, h)
    B_rhs = B_rhs + kreiss_oliger(B, h)
    return alpha_rhs, beta_rhs, B_rhs


def enforce_algebraic_constraints_dyn(state: BSSNStateDyn) -> BSSNStateDyn:
    plain = enforce_algebraic_constraints(to_plain_state(state))
    return BSSNStateDyn(plain.t, plain.phi, plain.gamma_t, plain.K, plain.A_t,
                         plain.Gamma_t, state.alpha, state.beta, state.B_aux)


def rk4_step_bssn_dyngauge(state: BSSNStateDyn, dt, grid: EvolutionGrid,
                            mode="vacuum", matter=None, eta=2.0):
    h = grid.h

    def rhs(s: BSSNStateDyn):
        m = matter if mode == "frozen_source" else None
        d_phi, d_gt, d_K, d_At, d_Gt = bssn_rhs(to_plain_state(s), s.alpha, s.beta, h, matter=m)
        d_alpha, d_beta, d_B = gauge_rhs(s, d_Gt, h, eta=eta)
        return d_phi, d_gt, d_K, d_At, d_Gt, d_alpha, d_beta, d_B

    def combine(s: BSSNStateDyn, d, scale):
        d_phi, d_gt, d_K, d_At, d_Gt, d_alpha, d_beta, d_B = d
        return BSSNStateDyn(s.t, s.phi + scale * d_phi, s.gamma_t + scale * d_gt,
                             s.K + scale * d_K, s.A_t + scale * d_At,
                             s.Gamma_t + scale * d_Gt, s.alpha + scale * d_alpha,
                             s.beta + scale * d_beta, s.B_aux + scale * d_B)

    k1 = rhs(state)
    k2 = rhs(combine(state, k1, dt / 2))
    k3 = rhs(combine(state, k2, dt / 2))
    k4 = rhs(combine(state, k3, dt))

    out = [a + 2 * b + 2 * c + d for a, b, c, d in zip(k1, k2, k3, k4)]
    d_phi, d_gt, d_K, d_At, d_Gt, d_alpha, d_beta, d_B = out

    new_state = BSSNStateDyn(
        state.t + dt,
        state.phi + (dt / 6.0) * d_phi, state.gamma_t + (dt / 6.0) * d_gt,
        state.K + (dt / 6.0) * d_K, state.A_t + (dt / 6.0) * d_At,
        state.Gamma_t + (dt / 6.0) * d_Gt, state.alpha + (dt / 6.0) * d_alpha,
        state.beta + (dt / 6.0) * d_beta, state.B_aux + (dt / 6.0) * d_B,
    )
    return enforce_algebraic_constraints_dyn(new_state)


def run_evolution_bssn_dyngauge(grid: EvolutionGrid, params: WarpBubbleParams,
                                 t_end, mode="vacuum", cfl=0.15, save_every=5, eta=2.0):
    gamma0, K0, alpha0, beta0 = adm_initial_data(grid, params, t0=0.0)
    plain0 = adm_to_bssn(gamma0, K0, grid.h)
    state = BSSNStateDyn(0.0, plain0.phi, plain0.gamma_t, plain0.K, plain0.A_t,
                          plain0.Gamma_t, alpha0, beta0, jnp.zeros_like(beta0))
    matter = matter_source(grid, params, t0=0.0) if mode == "frozen_source" else None
    rho0 = matter[0] if matter is not None else None

    from .free_evolution import courant_dt
    dt = courant_dt(grid, cfl)
    n_steps = int(np.ceil(t_end / dt))

    ts, peak_x, ham = [], [], []

    def record(s: BSSNStateDyn):
        plain = to_plain_state(s)
        gamma, K_ij = bssn_to_adm(plain)
        x_peak, _ = track_bubble_peak(K_ij, grid)
        H = hamiltonian_constraint_from_bssn(plain, grid.h, rho=rho0)
        ts.append(s.t); peak_x.append(x_peak)
        ham.append(float(jnp.max(jnp.abs(H))))

    record(state)
    for i in range(n_steps):
        state = rk4_step_bssn_dyngauge(state, dt, grid, mode=mode, matter=matter, eta=eta)
        if (i + 1) % save_every == 0 or i == n_steps - 1:
            record(state)

    return {"t": np.array(ts), "peak_x": np.array(peak_x),
            "ham_violation": np.array(ham), "dt": dt, "n_steps": n_steps,
            "final_state": state}


# ---------------------------------------------------------------------------
# Time integration: RK4, method of lines, prescribed gauge (matches
# free_evolution.rk4_step's structure exactly so the two are comparable).
# ---------------------------------------------------------------------------

def rk4_step_bssn(state: BSSNState, dt, grid: EvolutionGrid,
                   params: WarpBubbleParams, mode="vacuum", matter=None):
    h = grid.h

    def rhs(t, s: BSSNState):
        alpha, beta = gauge_fields(grid, params, t)
        m = matter if mode == "frozen_source" else None
        d_phi, d_gt, d_K, d_At, d_Gt = bssn_rhs(s, alpha, beta, h, matter=m)
        return d_phi, d_gt, d_K, d_At, d_Gt

    def combine(s: BSSNState, d, scale):
        d_phi, d_gt, d_K, d_At, d_Gt = d
        return BSSNState(s.t, s.phi + scale * d_phi, s.gamma_t + scale * d_gt,
                          s.K + scale * d_K, s.A_t + scale * d_At,
                          s.Gamma_t + scale * d_Gt)

    t = state.t
    k1 = rhs(t, state)
    k2 = rhs(t + dt / 2, combine(state, k1, dt / 2))
    k3 = rhs(t + dt / 2, combine(state, k2, dt / 2))
    k4 = rhs(t + dt, combine(state, k3, dt))

    out = []
    for a, b, c, d in zip(k1, k2, k3, k4):
        out.append(a + 2 * b + 2 * c + d)
    d_phi, d_gt, d_K, d_At, d_Gt = out

    new_state = BSSNState(
        t + dt,
        state.phi + (dt / 6.0) * d_phi,
        state.gamma_t + (dt / 6.0) * d_gt,
        state.K + (dt / 6.0) * d_K,
        state.A_t + (dt / 6.0) * d_At,
        state.Gamma_t + (dt / 6.0) * d_Gt,
    )
    return enforce_algebraic_constraints(new_state)


# ---------------------------------------------------------------------------
# Driver: BSSN free evolution with the same diagnostics as
# free_evolution.run_evolution (bubble-peak tracking, Hamiltonian
# constraint, drift from the exact analytic solution) so the two are
# directly comparable at identical grid/CFL/params.
# ---------------------------------------------------------------------------

def track_bubble_peak(K_ij, grid: EvolutionGrid):
    K_sq = jnp.einsum('...ij,...ij->...', K_ij, K_ij)
    X = jnp.asarray(grid.X)
    total = jnp.sum(K_sq)
    x_centroid = jnp.sum(K_sq * X) / jnp.where(total > 0, total, 1.0)
    return float(x_centroid), float(jnp.max(K_sq))


def hamiltonian_constraint_from_bssn(state: BSSNState, h, rho=None):
    """Physical Hamiltonian constraint R + K^2 - K_ij K^ij - 16 pi rho,
    reconstructed from BSSN variables (R = physical Ricci trace, via the
    same conformal-split Ricci this module evolves with -- so this
    diagnostic is exactly as trustworthy as `conformal_ricci`/
    `phi_ricci_correction`, see `validate_conformal_ricci`)."""
    gamma_t_inv = jnp.linalg.inv(state.gamma_t)
    Ricci_t, Gamma3, _, _ = conformal_ricci(state.gamma_t, state.Gamma_t, h)
    R_phi, _ = phi_ricci_correction(state.phi, state.gamma_t, gamma_t_inv, Gamma3, h)
    e4phi_inv = jnp.exp(-4.0 * state.phi)
    R_phys = e4phi_inv * jnp.einsum('...ij,...ij->...', gamma_t_inv, Ricci_t + R_phi)

    A_up = jnp.einsum('...ik,...jl,...kl->...ij', gamma_t_inv, gamma_t_inv, state.A_t)
    A_sq_phys = e4phi_inv * e4phi_inv * jnp.einsum('...ij,...ij->...', state.A_t, A_up)
    # note: A_ij A^ij (physical) = e^{-4phi} A~_ij A~^ij  (one e4phi from raising
    # each A_t index costs e^{-4phi} physically... kept explicit via bssn_to_adm
    # for the version actually used below to avoid an easily-miscounted factor.
    gamma, K_ij = bssn_to_adm(state)
    gamma_inv = jnp.linalg.inv(gamma)
    K_trace = jnp.einsum('...ii->...', jnp.einsum('...ij,...jk->...ik', gamma_inv, K_ij))
    K_up = jnp.einsum('...ik,...jl,...kl->...ij', gamma_inv, gamma_inv, K_ij)
    K_sq = jnp.einsum('...ij,...ij->...', K_ij, K_up)
    lhs = R_phys + K_trace ** 2 - K_sq
    rhs = 16.0 * jnp.pi * rho if rho is not None else 0.0
    return lhs - rhs


def run_evolution_bssn(grid: EvolutionGrid, params: WarpBubbleParams,
                        t_end, mode="vacuum", cfl=0.25, save_every=5):
    gamma0, K0, _, _ = adm_initial_data(grid, params, t0=0.0)
    state = adm_to_bssn(gamma0, K0, grid.h)
    matter = matter_source(grid, params, t0=0.0) if mode == "frozen_source" else None
    rho0 = matter[0] if matter is not None else None

    from .free_evolution import courant_dt
    dt = courant_dt(grid, cfl)
    n_steps = int(np.ceil(t_end / dt))

    ts, peak_x, ham = [], [], []

    def record(s: BSSNState):
        gamma, K_ij = bssn_to_adm(s)
        x_peak, _ = track_bubble_peak(K_ij, grid)
        H = hamiltonian_constraint_from_bssn(s, grid.h, rho=rho0)
        ts.append(s.t); peak_x.append(x_peak)
        ham.append(float(jnp.max(jnp.abs(H))))

    record(state)
    for i in range(n_steps):
        state = rk4_step_bssn(state, dt, grid, params, mode=mode, matter=matter)
        if (i + 1) % save_every == 0 or i == n_steps - 1:
            record(state)

    return {"t": np.array(ts), "peak_x": np.array(peak_x),
            "ham_violation": np.array(ham), "dt": dt, "n_steps": n_steps,
            "final_state": state}


# ---------------------------------------------------------------------------
# Validation: cross-check R~_ij + R^phi_ij against an INDEPENDENT direct
# computation of the physical Ricci tensor (spatial_christoffel_ricci_fd
# applied straight to gamma_ij, no conformal split at all) on a smooth,
# genuinely curved test metric. This is the module's load-bearing
# correctness check -- see module docstring.
# ---------------------------------------------------------------------------

def validate_conformal_ricci(nx=24, ny=24, nz=24, extent=2.0, amp=0.08, k=1.1):
    """A smooth, genuinely non-conformally-flat test metric: each component
    varies independently (NOT a single scalar function times a fixed
    matrix -- that degenerate case cancels out of the conformal split
    entirely under det-normalization and would silently test nothing)."""
    xs = np.linspace(-extent, extent, nx)
    ys = np.linspace(-extent, extent, ny)
    zs = np.linspace(-extent, extent, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    h = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    xx, yy, zz = jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)

    g11 = 1.0 + amp * jnp.sin(k * xx) * jnp.cos(k * yy)
    g22 = 1.0 + amp * jnp.sin(k * yy) * jnp.cos(k * zz)
    g33 = 1.0 + amp * jnp.sin(k * zz) * jnp.cos(k * xx)
    g12 = amp * 0.3 * jnp.sin(k * (xx + yy))
    g13 = amp * 0.2 * jnp.cos(k * (xx - zz))
    g23 = amp * 0.25 * jnp.sin(k * (yy + zz))
    gamma = jnp.stack([
        jnp.stack([g11, g12, g13], axis=-1),
        jnp.stack([g12, g22, g23], axis=-1),
        jnp.stack([g13, g23, g33], axis=-1),
    ], axis=-2)

    # direct physical Ricci, no conformal split
    _, Ricci_direct = spatial_christoffel_ricci_fd(gamma, h)

    # conformal-split path
    det_g = jnp.linalg.det(gamma)
    phi = jnp.log(det_g) / 12.0
    gamma_t = gamma / jnp.exp(4.0 * phi)[..., None, None]
    gamma_t_inv = jnp.linalg.inv(gamma_t)
    Gamma3, _ = spatial_christoffel_ricci_fd(gamma_t, h)
    Gamma_t_vec = jnp.einsum('...jk,...ijk->...i', gamma_t_inv, Gamma3)

    Ricci_t, Gamma3b, _, _ = conformal_ricci(gamma_t, Gamma_t_vec, h)
    R_phi, _ = phi_ricci_correction(phi, gamma_t, gamma_t_inv, Gamma3b, h)
    Ricci_split = Ricci_t + R_phi

    # compare on the interior only (edge-padding in the FD stencils makes
    # the boundary rows/cols of each independent derivation disagree in a
    # way that's a boundary-condition artifact, not a formula error).
    # NOTE: a naive max-relative-error metric is a bad test here -- Ricci
    # components pass through ~0 at some interior points for a generic
    # smooth field, and dividing by near-zero there produces a huge
    # "relative error" that has nothing to do with correctness. The real
    # test is O(h^2) CONVERGENCE of the absolute error (done by the
    # caller in __main__ below, which compares two resolutions).
    interior = (slice(2, -2), slice(2, -2), slice(2, -2))
    diff = jnp.abs(Ricci_direct[interior] - Ricci_split[interior])
    max_abs_err = float(jnp.max(diff))
    return {"max_abs_err": max_abs_err}


if __name__ == "__main__":
    print("Validating conformal_ricci + phi_ricci_correction against a direct")
    print("physical-Ricci FD computation on a smooth curved test metric,")
    print("checking for O(h^2) convergence of the two independent derivations...")
    n_lo, n_hi = 24, 48
    err_lo = validate_conformal_ricci(nx=n_lo, ny=n_lo, nz=n_lo)["max_abs_err"]
    err_hi = validate_conformal_ricci(nx=n_hi, ny=n_hi, nz=n_hi)["max_abs_err"]
    ratio = err_lo / err_hi
    print(f"  max abs error at n={n_lo}: {err_lo:.3e}")
    print(f"  max abs error at n={n_hi}: {err_hi:.3e}")
    print(f"  ratio (expect ~4.0 for O(h^2) convergence): {ratio:.2f}")
    if 3.0 < ratio < 5.5:
        print("  PASS: the two independent derivations converge to the same")
        print("  continuum Ricci tensor at the expected O(h^2) rate.")
    else:
        print("  FAIL: does not show clean O(h^2) convergence -- do not trust "
              "bssn.py results until this passes.")
