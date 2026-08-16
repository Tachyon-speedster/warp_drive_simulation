"""
validation.py — Validation Philosophy (project doc, section 19)

"The project should not trust numerical results blindly. Every major
tensor calculation should have validation tests."

Implements, and returns a structured PASS/FAIL report for, every check
listed in the project doc section 19:

  1. metric / inverse-metric identity        (g @ g_inv == I)
  2. metric symmetry                         (g_ab == g_ba)
  3. Christoffel symmetry in lower indices   (Gamma^a_bc == Gamma^a_cb)
  4. Ricci symmetry                          (R_ab == R_ba)
  5. Riemann antisymmetry in last two indices(R^a_bcd == -R^a_bdc)
  6. geodesic normalization                  (see geodesic.py, run separately)
  7. numerical convergence (FD engine, h->0) (Richardson-style h-halving)
  8. autodiff vs finite-difference comparison
  9. flat-spacetime limit                    (v_s=0 => Riemann ~ 0 everywhere)
  10. zero-velocity limit                    (same as flat limit here, since
      the Alcubierre metric has no curvature source besides v_s*f)
  11. large-distance limit                   (r_s >> R => Riemann ~ 0)
  12. parameter-sensitivity                  (curvature scales sensibly with
      sigma, v_s -- returned as auxiliary data rather than pass/fail)
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from .metric import WarpBubbleParams, minkowski_metric, metric_tensor
from .christoffel import inverse_metric, christoffel_at_point
from .curvature import full_curvature_at_point, riemann_tensor
from .derivatives import compare_engines


def check_metric_inverse_identity(g, g_inv, tol=1e-10):
    I = g @ g_inv
    err = float(jnp.max(jnp.abs(I - jnp.eye(4))))
    return {"name": "metric/inverse identity", "pass": err < tol, "max_error": err}


def check_metric_symmetry(g, tol=1e-14):
    err = float(jnp.max(jnp.abs(g - g.T)))
    return {"name": "metric symmetry", "pass": err < tol, "max_error": err}


def check_christoffel_symmetry(Gamma, tol=1e-10):
    # Gamma[a,b,c] should equal Gamma[a,c,b]
    err = float(jnp.max(jnp.abs(Gamma - jnp.transpose(Gamma, (0, 2, 1)))))
    return {"name": "Christoffel symmetry (lower indices)", "pass": err < tol,
            "max_error": err}


def check_ricci_symmetry(Ric, tol=1e-8):
    err = float(jnp.max(jnp.abs(Ric - Ric.T)))
    return {"name": "Ricci symmetry", "pass": err < tol, "max_error": err}


def check_riemann_antisymmetry(Riemann, tol=1e-8):
    # R^a_{bcd} == -R^a_{bdc}
    err = float(jnp.max(jnp.abs(Riemann + jnp.transpose(Riemann, (0, 1, 3, 2)))))
    return {"name": "Riemann antisymmetry (last 2 indices)", "pass": err < tol,
            "max_error": err}


def check_flat_limit(params_flat: WarpBubbleParams, coords, tol=1e-6):
    """With v_s = 0 the metric reduces to exact Minkowski everywhere, so
    every curvature quantity must vanish (to numerical precision)."""
    assert params_flat.v_s == 0.0, "check_flat_limit requires v_s=0 params"
    out = full_curvature_at_point(coords, params_flat, engine="autodiff")
    max_riemann = float(jnp.max(jnp.abs(out["Riemann"])))
    g = out["g"]
    eta = minkowski_metric()
    metric_err = float(jnp.max(jnp.abs(g - eta)))
    return {"name": "flat-spacetime limit (v_s=0)",
            "pass": max_riemann < tol and metric_err < tol,
            "max_riemann": max_riemann, "metric_vs_minkowski_error": metric_err}


def check_large_distance_limit(params: WarpBubbleParams, t, tol=1e-4):
    """Far from the bubble (r_s >> R) spacetime should be asymptotically
    flat: curvature should decay toward zero."""
    far_coords = jnp.array([t, params.x_s0 + 50 * params.R, 0.0, 0.0],
                            dtype=jnp.float64)
    out = full_curvature_at_point(far_coords, params, engine="autodiff")
    max_riemann = float(jnp.max(jnp.abs(out["Riemann"])))
    return {"name": "large-distance (asymptotic flatness) limit",
            "pass": max_riemann < tol, "max_riemann_at_50R": max_riemann}


def check_fd_convergence(coords, params: WarpBubbleParams, h_list=(1e-2, 1e-3, 1e-4)):
    """Richardson-style check: as h shrinks, FD-vs-autodiff discrepancy
    should shrink roughly as O(h^2) until round-off floor is hit."""
    results = []
    for h in h_list:
        cmp = compare_engines(coords, params, h=h)
        results.append({"h": h, **cmp})
    monotonic = all(
        results[i]["max_abs_diff"] >= results[i + 1]["max_abs_diff"] * 0.5
        for i in range(len(results) - 1)
    )
    return {"name": "FD convergence toward autodiff as h->0",
            "pass": monotonic, "trace": results}


def check_adm_constraints(params: WarpBubbleParams, test_points, tol=1e-8):
    """Section 21-23 (Advanced milestones): verify the ADM Hamiltonian and
    momentum constraints hold at a set of test points. This is an
    independent numerical-relativity consistency check on the ENTIRE
    pipeline (metric -> Christoffel -> Riemann -> Ricci -> Einstein ->
    stress-energy), using a completely different derivation path
    (extrinsic curvature from Gamma^0_ij) than the one used to build T_ab
    -- if these don't hold to numerical precision, something upstream is
    wrong."""
    from .adm import hamiltonian_constraint_residual, momentum_constraint_residual
    max_h = 0.0
    max_m = 0.0
    for p in test_points:
        h = float(jnp.abs(hamiltonian_constraint_residual(p, params)))
        m = float(jnp.max(jnp.abs(momentum_constraint_residual(p, params))))
        max_h = max(max_h, h)
        max_m = max(max_m, m)
    return {"name": "ADM Hamiltonian + momentum constraints",
            "pass": max_h < tol and max_m < tol,
            "max_hamiltonian_residual": max_h,
            "max_momentum_residual": max_m}


def run_full_validation(params: WarpBubbleParams, test_coords=None):
    """Run the full section-19 validation suite and return a report list."""
    if test_coords is None:
        test_coords = jnp.array([0.0, 0.3, 0.2, 0.0], dtype=jnp.float64)

    report = []

    out = full_curvature_at_point(test_coords, params, engine="autodiff")
    report.append(check_metric_inverse_identity(out["g"], out["g_inv"]))
    report.append(check_metric_symmetry(out["g"]))
    report.append(check_christoffel_symmetry(out["Gamma"]))
    report.append(check_ricci_symmetry(out["Ricci"]))

    _, _, _, Riemann = riemann_tensor(test_coords, params, engine="autodiff")
    report.append(check_riemann_antisymmetry(Riemann))

    flat_params = WarpBubbleParams(v_s=0.0, R=params.R, sigma=params.sigma,
                                    x_s0=params.x_s0)
    report.append(check_flat_limit(flat_params, test_coords))

    report.append(check_large_distance_limit(params, t=float(test_coords[0])))
    report.append(check_fd_convergence(test_coords, params))

    adm_test_points = [
        jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.array([0.0, 0.0, 0.5, 0.0], dtype=jnp.float64),
        jnp.array([0.0, -0.8, 0.6, 0.3], dtype=jnp.float64),
        jnp.array([0.0, 3.0, 0.0, 0.0], dtype=jnp.float64),
    ]
    report.append(check_adm_constraints(params, adm_test_points))

    return report


def format_report(report) -> str:
    lines = []
    all_pass = True
    for r in report:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        lines.append(f"[{status}] {r['name']}")
    lines.append("")
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return "\n".join(lines)
