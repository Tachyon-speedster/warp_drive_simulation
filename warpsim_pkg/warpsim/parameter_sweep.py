"""
parameter_sweep.py — Parameter Sweeps + Integrated Negative Energy
(project doc "Then" milestones 11-14)

For a given WarpBubbleParams, evaluates the energy-density field over a
2D grid and integrates the negative part (Riemann sum: sum(rho[rho<0]) *
dx * dy), giving a single scalar "how much exotic matter, integrated over
this cross-section, does this bubble configuration require" -- the
natural quantity to sweep over v_s / R / sigma and compare across bubble
configurations (milestone 14 & 15).

This is a 2D-slice integral (units: energy density * area), not a full 3D
volume integral -- correct 3D would integrate the same integrand over a
3D grid (x,y,z) rather than the z=0 slice used elsewhere in this project.
That's a straightforward extension (add a z-loop / 3D grid to grid.py) if
you need true 3D-integrated exotic-mass numbers; the relative trends
across parameter sweeps (which is usually what you want first) are
already meaningful from the 2D slice since the bubble is axially
symmetric about the direction of motion.
"""
from __future__ import annotations
import numpy as np

from .metric import WarpBubbleParams
from .grid import make_grid, evaluate_grid_fields, make_grid_3d


def integrated_negative_energy(params: WarpBubbleParams, x_range=(-6, 6),
                                y_range=(-6, 6), nx=100, ny=100, t=0.0):
    """Returns (integrated_negative_energy, max_abs_R_scalar, fields, X, Y)."""
    X, Y, coords = make_grid(x_range=x_range, y_range=y_range, nx=nx, ny=ny, t=t)
    fields = evaluate_grid_fields(coords, params, shape=X.shape)
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    rho = fields["energy_density"]
    neg_integral = float(np.sum(np.where(rho < 0, rho, 0.0)) * dx * dy)
    max_abs_R = float(np.max(np.abs(fields["R_scalar"])))
    return neg_integral, max_abs_R, fields, X, Y


def sweep_velocity(v_s_values, R=1.0, sigma=8.0, **grid_kwargs):
    """Sweep bubble coordinate velocity; R, sigma fixed."""
    results = []
    for v_s in v_s_values:
        params = WarpBubbleParams(v_s=float(v_s), R=R, sigma=sigma)
        neg_e, max_R, _, _, _ = integrated_negative_energy(params, **grid_kwargs)
        results.append({"v_s": float(v_s), "integrated_negative_energy": neg_e,
                         "max_abs_R_scalar": max_R})
    return results


def sweep_radius(R_values, v_s=2.0, sigma=8.0, **grid_kwargs):
    """Sweep bubble radius; v_s, sigma fixed."""
    results = []
    for R in R_values:
        params = WarpBubbleParams(v_s=v_s, R=float(R), sigma=sigma)
        neg_e, max_R, _, _, _ = integrated_negative_energy(params, **grid_kwargs)
        results.append({"R": float(R), "integrated_negative_energy": neg_e,
                         "max_abs_R_scalar": max_R})
    return results


def sweep_wall_thickness(sigma_values, v_s=2.0, R=1.0, **grid_kwargs):
    """Sweep wall steepness sigma (larger sigma = thinner wall);
    v_s, R fixed."""
    results = []
    for sigma in sigma_values:
        params = WarpBubbleParams(v_s=v_s, R=R, sigma=float(sigma))
        neg_e, max_R, _, _, _ = integrated_negative_energy(params, **grid_kwargs)
        results.append({"sigma": float(sigma), "integrated_negative_energy": neg_e,
                         "max_abs_R_scalar": max_R})
    return results


def integrated_negative_energy_3d(params: WarpBubbleParams,
                                   x_range=(-4, 4), y_range=(-3, 3),
                                   z_range=(-3, 3), nx=30, ny=24, nz=24,
                                   t=0.0):
    """True 3D volume integral of the exotic (negative) energy density --
    the correct completion of milestone 14 ('quantify integrated
    negative-energy regions'), vs. the 2D-slice proxy used by
    `integrated_negative_energy` above. Returns
    (integrated_negative_energy_3d, fields, coords, shape, dV)."""
    coords, shape, dV = make_grid_3d(x_range=x_range, y_range=y_range,
                                      z_range=z_range, nx=nx, ny=ny, nz=nz, t=t)
    fields = evaluate_grid_fields(coords, params, shape=shape)
    rho = fields["energy_density"]
    neg_integral = float(np.sum(np.where(rho < 0, rho, 0.0)) * dV)
    return neg_integral, fields, coords, shape, dV
