"""
run_demo.py — End-to-end exercise of the warpsim pipeline.

Produces:
  1. A validation report (section 19 checks).
  2. Autodiff-vs-finite-difference derivative comparison, in Python and
     cross-checked against the compiled C++ engine.
  3. 2D field maps: Ricci scalar, Einstein tensor G_tt, observer-measured
     energy density (section 20 investigation, milestones 1-5).
  4. A geodesic trajectory through the warp geometry with its
     normalization-error trace (sections 12-13).

Run:  python3 run_demo.py
"""
import time
import numpy as np
import jax.numpy as jnp

from warpsim.metric import WarpBubbleParams
from warpsim.derivatives import compare_engines
from warpsim.curvature import full_curvature_at_point
from warpsim.stress_energy import stress_energy_tensor
from warpsim.observer import normalize_eulerian_observer, normalize_static_observer, energy_density, is_timelike
from warpsim.geodesic import integrate_geodesic
from warpsim.grid import make_grid, evaluate_grid_fields
from warpsim.validation import run_full_validation, format_report
from warpsim.visualize import plot_field, plot_geodesic, plot_normalization_error
from warpsim.adm import adm_decompose

OUT = "outputs"

def main():
    params = WarpBubbleParams(v_s=2.0, R=1.0, sigma=8.0, x_s0=0.0)
    print("=" * 70)
    print("WARP-BUBBLE SPACETIME SIMULATOR — full pipeline demo")
    print(f"Bubble parameters: v_s={params.v_s}, R={params.R}, sigma={params.sigma}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Validation suite (section 19)
    # ---------------------------------------------------------------
    print("\n[1] Running validation suite (section 19)...")
    report = run_full_validation(params)
    print(format_report(report))

    # ---------------------------------------------------------------
    # 2. Autodiff vs finite-difference at the bubble wall (section 18)
    # ---------------------------------------------------------------
    print("\n[2] Autodiff vs finite-difference at the bubble wall...")
    wall_coords = jnp.array([0.0, params.R, 0.0, 0.0], dtype=jnp.float64)  # r_s = R
    for h in (1e-2, 1e-4, 1e-6):
        cmp = compare_engines(wall_coords, params, h=h)
        print(f"    h={h:>9.0e}   max_abs_diff={cmp['max_abs_diff']:.3e}   "
              f"max_rel_diff={cmp['max_rel_diff']:.3e}")
    print("    (autodiff is exact/h-independent; FD error should shrink ~h^2 "
          "then floor out on round-off)")

    # ---------------------------------------------------------------
    # 3. Full geometry at bubble center and wall + 3+1 split
    # ---------------------------------------------------------------
    print("\n[3] Full curvature pipeline + 3+1 (ADM) decomposition at bubble center...")
    center = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    out = full_curvature_at_point(center, params, engine="autodiff")
    alpha, beta_i, beta_up, gamma_ij = adm_decompose(out["g"])
    print(f"    lapse alpha = {float(alpha):.6f}  (Alcubierre gauge => should be ~1)")
    print(f"    shift beta^i = {np.asarray(beta_up)}")
    print(f"    Ricci scalar R = {float(out['R_scalar']):.6e}")

    T = stress_energy_tensor(out["Einstein"])
    u = normalize_eulerian_observer(out["g"])
    assert is_timelike(out["g"], u)
    rho = energy_density(T, u)
    print(f"    Eulerian-observer energy density T_ab u^a u^b = {float(rho):.6e}")

    # ---------------------------------------------------------------
    # 4. 2D field maps over the grid (sections 15-17, milestones 1-5)
    # ---------------------------------------------------------------
    print("\n[4] Evaluating 2D field maps over spatial grid (this JIT-compiles once)...")
    t0 = time.time()
    X, Y, coords_flat = make_grid(x_range=(-6, 6), y_range=(-5, 5), nx=140, ny=110, t=0.0)
    fields = evaluate_grid_fields(coords_flat, params, shape=X.shape)
    t1 = time.time()
    print(f"    grid: {X.shape[1]}x{X.shape[0]} points, autodiff pipeline, "
          f"wall time = {t1 - t0:.3f}s (includes JIT compile)")

    plot_field(X, Y, fields["R_scalar"], "Ricci Scalar R(x,y), t=0",
               "R (curvature scalar)", f"{OUT}/ricci_scalar_map.png",
               contour_bubble=(params.x_s0, params.R))
    plot_field(X, Y, fields["G_tt"], "Einstein Tensor Component G_tt(x,y), t=0",
               "G_tt", f"{OUT}/einstein_Gtt_map.png",
               contour_bubble=(params.x_s0, params.R))
    plot_field(X, Y, fields["energy_density"],
               "Observer-Measured Energy Density rho = T_ab u^a u^b (Eulerian observer)",
               "rho (negative = exotic matter required)",
               f"{OUT}/energy_density_map.png",
               contour_bubble=(params.x_s0, params.R))
    print(f"    saved: {OUT}/ricci_scalar_map.png, {OUT}/einstein_Gtt_map.png, "
          f"{OUT}/energy_density_map.png")
    neg_frac = float(np.mean(fields["energy_density"] < 0))
    print(f"    fraction of grid with rho < 0 (energy-condition-violating region): "
          f"{neg_frac:.3%}")

    # ---------------------------------------------------------------
    # 5. Geodesic integration (sections 12-13)
    # ---------------------------------------------------------------
    print("\n[5] Integrating a massive-particle geodesic through the warp geometry...")
    # Start just outside the wall, off-axis, so the bubble (moving at v_s=2
    # along +x) sweeps past the particle during the integration window --
    # this is what actually exercises the curvature (a particle that never
    # gets near the wall would trivially stay at rest, which is correct
    # physics but a boring plot).
    x0 = jnp.array([0.0, -0.2, 1.3, 0.0], dtype=jnp.float64)
    g0 = full_curvature_at_point(x0, params, engine="autodiff")["g"]
    u0 = normalize_eulerian_observer(g0)  # starts co-moving with the local geometry
    result = integrate_geodesic(params, x0, u0, tau_span=(0.0, 2.0), n_eval=500)
    print(f"    integrator success: {result['success']}  ({result['message']})")
    print(f"    final position [t,x,y,z] = {result['final_position']}")
    print(f"    max |normalization drift| over trajectory = "
          f"{result['max_normalization_error']:.3e}")

    plot_geodesic(result["position"], params, f"{OUT}/geodesic_trajectory.png",
                  contour_bubble=(params.x_s0, params.R))
    plot_normalization_error(result["tau"], result["normalization_error"],
                              f"{OUT}/geodesic_normalization_error.png")
    print(f"    saved: {OUT}/geodesic_trajectory.png, "
          f"{OUT}/geodesic_normalization_error.png")

    # ---------------------------------------------------------------
    # 6. C++ engine cross-check (if built)
    # ---------------------------------------------------------------
    print("\n[6] Cross-checking against compiled C++ finite-difference engine...")
    try:
        from warpsim.cpp_bridge import evaluate_grid_fd_cpp, HAVE_CPP_ENGINE
        if not HAVE_CPP_ENGINE:
            raise ImportError("not built")
        small_X, small_Y, small_coords = make_grid(x_range=(-3, 3), y_range=(-3, 3),
                                                     nx=25, ny=25, t=0.0)
        t0 = time.time()
        cpp_fields = evaluate_grid_fd_cpp(np.asarray(small_coords), params,
                                           h_outer=1e-4, h_inner=1e-4)
        t_cpp = time.time() - t0

        py_fields = evaluate_grid_fields(small_coords, params, shape=small_X.shape)
        r_scalar_ad = py_fields["R_scalar"].ravel()
        r_scalar_cpp = cpp_fields["R_scalar"]
        diff = np.abs(r_scalar_ad - r_scalar_cpp)
        print(f"    C++ FD engine grid ({small_X.size} pts) wall time: {t_cpp:.4f}s")
        print(f"    max |R_scalar(autodiff) - R_scalar(C++ FD)| over grid: "
              f"{np.max(diff):.3e}")
        print(f"    mean |R_scalar(autodiff) - R_scalar(C++ FD)| over grid: "
              f"{np.mean(diff):.3e}")
        print("    (small residual differences here are FD truncation error, "
              "concentrated at/near the wall -- exactly the discrepancy "
              "autodiff was introduced to eliminate)")
    except ImportError:
        print("    C++ extension not built yet -- run cpp/build.sh first. Skipping.")

    print("\n" + "=" * 70)
    print("Demo complete. See outputs/ for all generated figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
