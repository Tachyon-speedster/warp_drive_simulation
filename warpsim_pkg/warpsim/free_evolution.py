"""
free_evolution.py -- Genuine time-evolving 3+1 numerical relativity

WHY THIS MODULE EXISTS
-----------------------
Every other module (metric.py ... adm_evolution.py) works with the
Alcubierre metric's *closed form*: g_{ab}(t,x,y,z) is known analytically
for every t, so "time" so far has only ever meant "plug a different t
into the same formula". adm_evolution.py goes one step further and
checks that this closed form *satisfies* the ADM evolution equation, but
it still never asks Einstein's equations to *predict* gamma_ij, K_ij at a
later time from data at an earlier time -- everything is still read off
the exact solution. That predictive step is what "time-evolving 3+1
spacetime" means in numerical relativity, and it is exactly the gap the
README flags under "Known next milestones".

This module builds a real (if intentionally minimal) free-evolution NR
code:

  1. ADM initial data (gamma_ij, K_ij, alpha, beta^i at t=0) is taken from
     the existing, already-validated autodiff pipeline (metric.py/adm.py)
     -- the initial slice is still exact GR.
  2. gamma_ij(x,y,z) and K_ij(x,y,z) are discretized on a finite 3D grid
     and become genuine dynamical variables (no longer forced to equal
     delta_ij / the closed form).
  3. They are advanced forward in time with the full nonlinear nonlinear
     ADM evolution equations (same RHS as adm_evolution.kij_evolution_rhs,
     generalized to a non-flat, finite-differenced spatial metric) via
     explicit 4th-order Runge-Kutta (method of lines), the standard
     approach used by real NR codes (e.g. the Einstein Toolkit / BSSN
     codes -- see MATH.md for how this relates to full BSSN).
  4. Constraint violation and drift from the exact analytic solution are
     tracked at every step -- the standard NR correctness/stability
     diagnostics.

WHAT THIS ANSWERS PHYSICALLY -- "can the warp move through spacetime?"
------------------------------------------------------------------------
The Alcubierre solution is an *exact* solution of G_ab = 8*pi*T_ab only if
T_ab is, at every instant, precisely the (generically negative-energy)
distribution metric.py's closed form implies. A real drive isn't a
spacetime you get to simply declare; it's a spacetime that must be
*sourced* by matter you actually control. This module lets that question
be asked numerically, with two evolution modes:

  * "vacuum"        -- T_ab = 0 for t>0 (the sourcing matter is switched
                        off the instant evolution starts).
  * "frozen_source"  -- the t=0 matter distribution (rho, S_ij) is held
                        fixed in the local Eulerian frame while gamma_ij,
                        K_ij evolve (matter present, but not continuously
                        re-engineered to track the moving bubble).

Neither mode re-derives T_ab from a real matter/field evolution equation
(that would require choosing and evolving an actual exotic-matter model,
a further extension noted in MATH.md), so neither is expected to hold the
bubble shape indefinitely. What they *do* show, and what makes this a
genuine test rather than a restatement of the analytic solution: whether
the region of large curvature / large negative energy density still
*translates* through the grid at coordinate speed v_s (see
`track_bubble_peak` below) even as gravity's own dynamics reshape it, and
how fast the ADM constraints and the deviation from the exact solution
grow. That is the real, dynamical version of "does the warp move through
spacetime" -- as opposed to the kinematic fact (already true by
construction of metric.py) that the *prescribed* metric's wall follows
x_s(t) = x_s0 + v_s t.

GAUGE
-----
alpha(t,x,y,z) and beta^i(t,x,y,z) are *not* evolved dynamically -- they
are re-evaluated at every RK substage directly from the closed-form
Alcubierre prescription (metric.py). Physically: alpha/beta are the drive's
control inputs. A real engineered warp field is exactly an externally
imposed lapse/shift, so holding them at their design values and asking
whether gamma_ij, K_ij *dynamically follow* is the meaningful test here.
(Fully free gauge -- 1+log slicing, Gamma-driver shift -- is a natural
extension for a future milestone; see MATH.md.)

FINITE DIFFERENCING
--------------------
Spatial derivatives use 2nd-order central differences with 'edge'
(zero-gradient) padding at the domain boundary, validated against the
existing autodiff spatial-Ricci computation (adm_evolution.spatial_ricci)
on a smooth conformally-flat test metric, confirming standard O(h^2)
convergence before being trusted on the bubble itself. 5-point
Kreiss-Oliger dissipation is added to every evolved field to damp the
grid-Nyquist noise that centered schemes are known to grow under
advection-dominated RHS terms (beta^k d_k K_ij is exactly such a term
here) -- standard practice in real NR codes, kept at low order to match
this project's existing "legacy finite-difference engine kept for
validation" philosophy (cpp/fd_engine.cpp) rather than inventing a new
high-order scheme.

RESOLUTION REQUIREMENT
------------------------
The Alcubierre wall has coordinate thickness ~ 1/sigma. Free evolution is
only meaningful if the grid resolves it (a handful of points per 1/sigma
at minimum); under-resolved grids will show large, non-convergent
constraint violation from step 1. `run_free_evolution.py` defaults to a
gentler wall (sigma=3) than the rest of the codebase's default (sigma=8)
specifically so a modest, laptop-friendly grid resolves it -- this is a
numerical-resolution choice, not a physics change.
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .metric import WarpBubbleParams, metric_tensor
from .adm import adm_decompose, extrinsic_curvature


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvolutionGrid:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    X: np.ndarray   # (nx,ny,nz)
    Y: np.ndarray
    Z: np.ndarray

    @property
    def h(self):
        return (self.dx, self.dy, self.dz)

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)


def make_evolution_grid(x_range=(-4.0, 6.0), y_range=(-4.0, 4.0),
                         z_range=(-4.0, 4.0), nx=56, ny=40, nz=40):
    xs = np.linspace(*x_range, nx)
    ys = np.linspace(*y_range, ny)
    zs = np.linspace(*z_range, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]; dz = zs[1] - zs[0]
    return EvolutionGrid(nx, ny, nz, dx, dy, dz, X, Y, Z)


# ---------------------------------------------------------------------------
# Finite-difference primitives. First three axes of any field are always
# (x,y,z); trailing axes are tensor indices carried along untouched.
# ---------------------------------------------------------------------------

def d_axis(f, h, axis):
    """2nd-order central first derivative along spatial axis 0/1/2, with
    edge-padding (zero-gradient) at the domain boundary."""
    pad_width = [(1, 1)] * 3 + [(0, 0)] * (f.ndim - 3)
    fp = jnp.pad(f, pad_width, mode="edge")
    sl_hi = [slice(1, -1)] * 3 + [slice(None)] * (f.ndim - 3)
    sl_lo = [slice(1, -1)] * 3 + [slice(None)] * (f.ndim - 3)
    sl_hi[axis] = slice(2, None)
    sl_lo[axis] = slice(0, -2)
    return (fp[tuple(sl_hi)] - fp[tuple(sl_lo)]) / (2.0 * h)


def grad3(f, h):
    """Stack of spatial derivatives: grad3(f)[a,...] = d_a f, a=0,1,2."""
    return jnp.stack([d_axis(f, h[a], a) for a in range(3)], axis=0)


def kreiss_oliger(f, h, eps=0.15):
    """-eps/16h * (f_{-2} - 4f_{-1} + 6f_0 - 4f_{+1} + f_{+2}) summed over
    the three spatial axes; standard grid-Nyquist damping."""
    out = jnp.zeros_like(f)
    for axis, h_ in zip((0, 1, 2), h):
        pad_width = [(2, 2)] * 3 + [(0, 0)] * (f.ndim - 3)
        fp = jnp.pad(f, pad_width, mode="edge")

        def sl(k):
            s = [slice(2, 2 + f.shape[a]) for a in range(3)]
            s[axis] = slice(2 + k, 2 + k + f.shape[axis])
            return tuple(s) + (slice(None),) * (f.ndim - 3)

        stencil = fp[sl(-2)] - 4 * fp[sl(-1)] + 6 * fp[sl(0)] - 4 * fp[sl(1)] + fp[sl(2)]
        out = out + (-eps / (16.0 * h_)) * stencil
    return out


# ---------------------------------------------------------------------------
# Spatial (3)Christoffel / (3)Ricci of the evolving gamma_ij, purely by
# finite differences (validated by convergence test against the existing
# autodiff spatial-Ricci implementation on a smooth test metric -- see
# tests/ or module changelog note; O(h^2) convergence confirmed).
# ---------------------------------------------------------------------------

def spatial_christoffel_ricci_fd(gamma, h):
    """gamma: (...,3,3) with leading spatial axes. Returns
    (Gamma3[...,a,b,c], Ricci3[...,b,d])."""
    shape3 = gamma.shape[:-2]
    P = int(np.prod(shape3))
    gamma_inv = jnp.linalg.inv(gamma)

    dgamma = grad3(gamma, h)                       # (3,...,m,n) = d_e gamma_mn
    dgamma2 = jnp.moveaxis(dgamma, 0, -1)           # (...,m,n,e)

    gamma_flat = gamma.reshape(P, 3, 3)
    gamma_inv_flat = gamma_inv.reshape(P, 3, 3)
    dgamma2_flat = dgamma2.reshape(P, 3, 3, 3)      # [p,m,n,e]

    term1 = jnp.einsum('pad,pdcb->pabc', gamma_inv_flat, dgamma2_flat)
    term2 = jnp.einsum('pad,pdbc->pabc', gamma_inv_flat, dgamma2_flat)
    term3 = jnp.einsum('pad,pbcd->pabc', gamma_inv_flat, dgamma2_flat)
    Gamma3_flat = 0.5 * (term1 + term2 - term3)     # [p,a,b,c] = Gamma^a_bc
    Gamma3 = Gamma3_flat.reshape(shape3 + (3, 3, 3))

    dGamma3 = grad3(Gamma3, h)                      # (3,...,a,b,c) = d_e Gamma^a_bc
    dGamma3_flat = dGamma3.reshape(3, P, 3, 3, 3)    # [e,p,a,b,d/c]

    TermA = jnp.einsum('cpabd->pabcd', dGamma3_flat)  # d_c Gamma^a_bd
    TermB = jnp.einsum('dpabc->pabcd', dGamma3_flat)  # d_d Gamma^a_bc
    GG1 = jnp.einsum('pace,pebd->pabcd', Gamma3_flat, Gamma3_flat)
    GG2 = jnp.einsum('pade,pebc->pabcd', Gamma3_flat, Gamma3_flat)
    Riemann3 = TermA - TermB + GG1 - GG2             # [p,a,b,c,d]
    Ricci3_flat = jnp.einsum('pabad->pbd', Riemann3)
    Ricci3 = Ricci3_flat.reshape(shape3 + (3, 3))
    return Gamma3, Ricci3


# ---------------------------------------------------------------------------
# Initial data & prescribed gauge, from the existing exact autodiff
# pipeline, vmapped across the grid.
# ---------------------------------------------------------------------------

def _flat_coords(grid: EvolutionGrid, t):
    fx, fy, fz = grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()
    return jnp.asarray(np.stack(
        [np.full_like(fx, t), fx, fy, fz], axis=1), dtype=jnp.float64)


def initial_data(grid: EvolutionGrid, params: WarpBubbleParams, t0=0.0):
    coords = _flat_coords(grid, t0)

    def point(c):
        g = metric_tensor(c, params)
        alpha, _, beta_up, gamma_ij = adm_decompose(g)
        K = extrinsic_curvature(c, params, engine="autodiff")
        return gamma_ij, K, alpha, beta_up

    gamma_f, K_f, alpha_f, beta_f = jax.vmap(point)(coords)
    s = grid.shape
    return (gamma_f.reshape(s + (3, 3)), K_f.reshape(s + (3, 3)),
            alpha_f.reshape(s), beta_f.reshape(s + (3,)))


def gauge_fields(grid: EvolutionGrid, params: WarpBubbleParams, t):
    """The drive's control inputs, re-evaluated exactly at time t."""
    coords = _flat_coords(grid, t)

    def point(c):
        g = metric_tensor(c, params)
        alpha, _, beta_up, _ = adm_decompose(g)
        return alpha, beta_up

    alpha_f, beta_f = jax.vmap(point)(coords)
    s = grid.shape
    return alpha_f.reshape(s), beta_f.reshape(s + (3,))


def matter_source(grid: EvolutionGrid, params: WarpBubbleParams, t0=0.0):
    """rho, S_ij, S measured by the Eulerian observer at t0 (used by the
    'frozen_source' evolution mode -- see module docstring)."""
    from .curvature import full_curvature_at_point
    from .stress_energy import stress_energy_tensor
    from .observer import normalize_eulerian_observer, energy_density

    coords = _flat_coords(grid, t0)

    def point(c):
        out = full_curvature_at_point(c, params, engine="autodiff")
        T = stress_energy_tensor(out["Einstein"])
        g = out["g"]
        n = normalize_eulerian_observer(g, check=False)
        rho = energy_density(T, n)
        n_lower = g @ n
        proj = jnp.eye(4, dtype=g.dtype) + jnp.outer(n, n_lower)
        S_full = jnp.einsum("ca,db,cd->ab", proj, proj, T)
        S_ij = S_full[1:4, 1:4]
        gamma_ij = g[1:4, 1:4]
        S = jnp.einsum("ij,ij->", jnp.linalg.inv(gamma_ij), S_ij)
        return rho, S_ij, S

    rho_f, S_ij_f, S_f = jax.vmap(point)(coords)
    s = grid.shape
    return rho_f.reshape(s), S_ij_f.reshape(s + (3, 3)), S_f.reshape(s)


# ---------------------------------------------------------------------------
# The ADM evolution RHS (finite-difference analogue of
# adm_evolution.kij_evolution_rhs, generalized to non-flat gamma_ij).
# ---------------------------------------------------------------------------

def adm_rhs(gamma, K, alpha, beta, h, matter=None):
    """matter: optional (rho, S_ij, S) tuple, same shape convention as
    gamma/K/alpha; if None, vacuum (T_ab=0)."""
    gamma_inv = jnp.linalg.inv(gamma)
    Gamma3, Ricci3 = spatial_christoffel_ricci_fd(gamma, h)

    beta_low = jnp.einsum('...jk,...k->...j', gamma, beta)
    d_i_beta_low_j = jnp.moveaxis(grad3(beta_low, h), 0, -2)      # (...,i,j)=d_i beta_j
    Gamma3_beta = jnp.einsum('...kij,...k->...ij', Gamma3, beta_low)
    D_i_beta_j = d_i_beta_low_j - Gamma3_beta
    dgamma_dt = (-2.0 * alpha[..., None, None] * K
                 + D_i_beta_j + jnp.swapaxes(D_i_beta_j, -1, -2))

    dK = grad3(K, h)                                              # (3,...,i,j)=d_a K_ij
    advection = jnp.einsum('a...,a...ij->...ij', jnp.moveaxis(beta, -1, 0), dK)

    d_j_beta_up_k = jnp.moveaxis(grad3(beta, h), 0, -2)           # (...,j,k)=d_j beta^k
    shift_coupling = (jnp.einsum('...ik,...jk->...ij', K, d_j_beta_up_k)
                       + jnp.einsum('...kj,...ik->...ij', K, d_j_beta_up_k))

    grad_alpha = grad3(alpha, h)                                  # (3,...)
    hess_alpha_axes = jnp.stack(
        [d_axis(grad_alpha[a], h[b], b) for a in range(3) for b in range(3)], axis=0
    ).reshape(3, 3, *alpha.shape)
    hess_alpha = jnp.moveaxis(hess_alpha_axes, (0, 1), (-2, -1))
    D2alpha = hess_alpha - jnp.einsum('...kij,...k->...ij', Gamma3, jnp.moveaxis(grad_alpha, 0, -1))

    K_trace = jnp.einsum('...ii->...', K)
    KK = jnp.einsum('...ik,...kl,...lj->...ij', K, gamma_inv, K)

    rhs_K = (advection + shift_coupling - D2alpha
             + alpha[..., None, None] * (Ricci3 + K_trace[..., None, None] * K - 2.0 * KK))

    if matter is not None:
        rho, S_ij, S = matter
        matter_term = S_ij - 0.5 * gamma * (S - rho)[..., None, None]
        rhs_K = rhs_K - 8.0 * jnp.pi * alpha[..., None, None] * matter_term

    rhs_K = rhs_K + kreiss_oliger(K, h)
    dgamma_dt = dgamma_dt + kreiss_oliger(gamma, h)
    return dgamma_dt, rhs_K


# ---------------------------------------------------------------------------
# Hamiltonian constraint (finite-difference version, for evolved fields
# where no closed form is available any more).
# ---------------------------------------------------------------------------

def hamiltonian_constraint_fd(gamma, K, h, rho=None):
    gamma_inv = jnp.linalg.inv(gamma)
    _, Ricci3 = spatial_christoffel_ricci_fd(gamma, h)
    R3 = jnp.einsum('...ij,...ij->...', gamma_inv, Ricci3)
    K_trace = jnp.einsum('...ii->...', K)
    K_up = jnp.einsum('...ik,...jl,...kl->...ij', gamma_inv, gamma_inv, K)
    K_sq = jnp.einsum('...ij,...ij->...', K, K_up)
    lhs = R3 + K_trace ** 2 - K_sq
    rhs = 16.0 * jnp.pi * rho if rho is not None else 0.0
    return lhs - rhs


# ---------------------------------------------------------------------------
# Time integration: explicit 4th-order Runge-Kutta, method of lines.
# alpha/beta are re-evaluated exactly at every substage (prescribed gauge).
# ---------------------------------------------------------------------------

@dataclass
class EvolutionState:
    t: float
    gamma: jnp.ndarray
    K: jnp.ndarray


def rk4_step(state: EvolutionState, dt, grid: EvolutionGrid,
             params: WarpBubbleParams, mode="vacuum", matter=None):
    h = grid.h

    def rhs(t, gamma, K):
        alpha, beta = gauge_fields(grid, params, t)
        m = matter if mode == "frozen_source" else None
        return adm_rhs(gamma, K, alpha, beta, h, matter=m)

    t, gamma, K = state.t, state.gamma, state.K
    k1g, k1K = rhs(t, gamma, K)
    k2g, k2K = rhs(t + dt / 2, gamma + dt / 2 * k1g, K + dt / 2 * k1K)
    k3g, k3K = rhs(t + dt / 2, gamma + dt / 2 * k2g, K + dt / 2 * k2K)
    k4g, k4K = rhs(t + dt, gamma + dt * k3g, K + dt * k3K)

    gamma_new = gamma + (dt / 6.0) * (k1g + 2 * k2g + 2 * k3g + k4g)
    K_new = K + (dt / 6.0) * (k1K + 2 * k2K + 2 * k3K + k4K)
    return EvolutionState(t + dt, gamma_new, K_new)


def courant_dt(grid: EvolutionGrid, cfl=0.25):
    return cfl * min(grid.dx, grid.dy, grid.dz)


# ---------------------------------------------------------------------------
# Diagnostics: does the bubble actually move through the evolving
# spacetime, and how fast do constraints/drift grow?
# ---------------------------------------------------------------------------

def track_bubble_peak(K, grid: EvolutionGrid):
    """K_ij K^ij-weighted centroid x-coordinate. The Alcubierre bubble's
    curvature/exotic-energy is concentrated in a thin *shell* at r_s ~ R
    (K ~ 0 at the very center, where f is locally flat), so many grid
    points share comparable K_ij K^ij -- a bare argmax jitters between
    near-degenerate shell points under discretization noise. The
    intensity-weighted centroid is the physically meaningful, robust
    stand-in for "where the bubble is" and converges to x_s(t) for a
    symmetric shell."""
    K_sq = jnp.einsum('...ij,...ij->...', K, K)
    X = jnp.asarray(grid.X)
    total = jnp.sum(K_sq)
    x_centroid = jnp.sum(K_sq * X) / jnp.where(total > 0, total, 1.0)
    return float(x_centroid), float(jnp.max(K_sq))


def drift_from_exact(gamma, K, grid: EvolutionGrid, params: WarpBubbleParams, t):
    gamma_exact, K_exact, _, _ = initial_data(grid, params, t0=t)
    return (float(jnp.max(jnp.abs(gamma - gamma_exact))),
            float(jnp.max(jnp.abs(K - K_exact))))


def run_evolution(grid: EvolutionGrid, params: WarpBubbleParams,
                   t_end, mode="vacuum", cfl=0.25, save_every=5):
    """Drives the full free evolution and returns a diagnostics dict of
    time series: t, bubble peak x-position, Hamiltonian constraint
    violation, and drift from the exact analytic solution."""
    gamma0, K0, _, _ = initial_data(grid, params, t0=0.0)
    matter = matter_source(grid, params, t0=0.0) if mode == "frozen_source" else None
    rho0 = matter[0] if matter is not None else None

    state = EvolutionState(0.0, gamma0, K0)
    dt = courant_dt(grid, cfl)
    n_steps = int(np.ceil(t_end / dt))

    ts, peak_x, ham, drift_g, drift_K = [], [], [], [], []

    def record(s: EvolutionState):
        x_peak, _ = track_bubble_peak(s.K, grid)
        H = hamiltonian_constraint_fd(s.gamma, s.K, grid.h, rho=rho0)
        dg, dK = drift_from_exact(s.gamma, s.K, grid, params, s.t)
        ts.append(s.t); peak_x.append(x_peak)
        ham.append(float(jnp.max(jnp.abs(H))))
        drift_g.append(dg); drift_K.append(dK)

    record(state)
    for i in range(n_steps):
        state = rk4_step(state, dt, grid, params, mode=mode, matter=matter)
        if (i + 1) % save_every == 0 or i == n_steps - 1:
            record(state)

    return {
        "t": np.array(ts), "peak_x": np.array(peak_x),
        "ham_violation": np.array(ham),
        "drift_gamma": np.array(drift_g), "drift_K": np.array(drift_K),
        "dt": dt, "n_steps": n_steps, "final_state": state,
    }
