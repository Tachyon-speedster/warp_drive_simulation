"""
grid.py — Spatial Grid / Field Engine + Generic Field Evaluator
(project doc, sections 15 & 16)

A 2D spatial cross-section (z=0, fixed t) is the default working surface,
matching the project doc's stated typical grid: x in [-10,10], y in [-6,6].

The generic field evaluator `evaluate_field_on_grid` takes any function
`point_fn(coords) -> scalar` (or -> small array, e.g. an Einstein-tensor
component) and vmaps it over every grid point, so the exact same
infrastructure produces the Ricci-scalar map, the energy-density map, any
single Einstein-tensor component map, etc., without bespoke grid code for
each (this is the point of the doc's section 16).

For speed, the whole geometry pipeline (metric -> Christoffel -> Riemann ->
Ricci -> Einstein -> stress-energy -> observer energy density) is written
as ONE jax-jittable function per grid point and then `jax.vmap`-ed across
the entire grid in a single compiled call, rather than looping in Python.
On a CPU-only box this still comfortably handles grids of a few hundred x a
few hundred points in well under a second after the first (compile) call.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from .metric import WarpBubbleParams
from .curvature import full_curvature_at_point
from .stress_energy import stress_energy_tensor
from .observer import normalize_eulerian_observer, energy_density
from .energy_conditions import check_nec


def make_grid(x_range=(-10.0, 10.0), y_range=(-6.0, 6.0), nx=120, ny=80,
              t=0.0, z=0.0):
    """Build a 2D (x,y) grid at fixed (t,z). Returns X, Y meshgrids (shape
    (ny,nx)) plus the flat coordinate array of shape (ny*nx, 4) in
    [t,x,y,z] order, ready for vmap."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(xs, ys)  # shape (ny, nx)
    flat_x = X.ravel()
    flat_y = Y.ravel()
    coords = np.stack([
        np.full_like(flat_x, t),
        flat_x,
        flat_y,
        np.full_like(flat_x, z),
    ], axis=1)  # (N,4)
    return X, Y, jnp.asarray(coords, dtype=jnp.float64)


def _single_point_fields(coords, params: WarpBubbleParams):
    """All diagnostic scalar/tensor fields at one spacetime point, using
    the exact autodiff pipeline throughout. Returns a dict of jax scalars
    suitable for vmap (fixed-shape outputs only)."""
    out = full_curvature_at_point(coords, params, engine="autodiff")
    T = stress_energy_tensor(out["Einstein"])
    u_obs = normalize_eulerian_observer(out["g"], check=False)
    rho = energy_density(T, u_obs)
    nec_min, _ = check_nec(T, out["g"], u_obs, n_theta=12)
    return {
        "R_scalar": out["R_scalar"],
        "G_tt": out["Einstein"][0, 0],
        "G_tx": out["Einstein"][0, 1],
        "T_tt": T[0, 0],
        "energy_density": rho,
        "nec_min": nec_min,
    }


def evaluate_grid_fields(coords_flat, params: WarpBubbleParams, shape):
    """Vectorized evaluation of `_single_point_fields` over every row of
    coords_flat (N,4), reshaped back to `shape` for each field.
    This is the "Generic Field Evaluator" of section 16: swap in any
    per-point function and get a field map for free. `shape` can be 2D
    (ny,nx) for a slice or 3D (nz,ny,nx) for a full volume (see
    `make_grid_3d` below) -- vmap doesn't care about the eventual reshape."""
    fn = lambda c: _single_point_fields(c, params)
    batched = jax.vmap(fn)(coords_flat)
    return {k: np.asarray(v).reshape(shape) for k, v in batched.items()}


def make_grid_3d(x_range=(-6.0, 6.0), y_range=(-6.0, 6.0), z_range=(-6.0, 6.0),
                  nx=40, ny=40, nz=40, t=0.0):
    """True 3D spatial grid (section 20/milestone 14: 'quantify integrated
    negative-energy regions' properly needs a volume, not a 2D slice).
    Returns the flat coordinate array (N,4) plus the grid shape (nz,ny,nx)
    and the cell volume dV, ready for `evaluate_grid_fields` + a Riemann-
    sum volume integral."""
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    zs = np.linspace(z_range[0], z_range[1], nz)
    X3, Y3, Z3 = np.meshgrid(xs, ys, zs, indexing="ij")  # each (nx,ny,nz)
    flat_x = X3.ravel(); flat_y = Y3.ravel(); flat_z = Z3.ravel()
    coords = np.stack([
        np.full_like(flat_x, t), flat_x, flat_y, flat_z,
    ], axis=1)
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    dz = (z_range[1] - z_range[0]) / (nz - 1)
    dV = dx * dy * dz
    return jnp.asarray(coords, dtype=jnp.float64), (nx, ny, nz), dV
