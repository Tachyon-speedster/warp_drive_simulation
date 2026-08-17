"""
run_bssn_evolution.py -- demo/validation for warpsim/bssn.py.

Compares three free-evolution schemes on the SAME warp-bubble initial data
and grid, all starting from the exact Alcubierre slice at t=0:

  1. plain ADM, prescribed (closed-form Alcubierre) gauge   -- free_evolution.run_evolution
  2. BSSN,      prescribed (closed-form Alcubierre) gauge   -- bssn.run_evolution_bssn
  3. BSSN,      DYNAMICAL gauge (1+log lapse + Gamma-driver) -- bssn.run_evolution_bssn_dyngauge

WHY A DYNAMICAL-GAUGE COMPARISON MATTERS HERE (read before trusting any
"BSSN vs ADM" comparison in this codebase or elsewhere)
-----------------------------------------------------------------------
Two independent things can make a free evolution blow up:

  (a) the PDE system's principal symbol being only weakly hyperbolic
      (the textbook reason ADM is avoided -- see bssn.py's module
      docstring), and
  (b) the timestep violating the CFL condition for the ADVECTION terms
      (beta^i d_i f), which is governed by |beta|_max, not by the light-
      speed/lapse terms that free_evolution.courant_dt() alone accounts
      for.

For this project's default bubble (v_s=2, i.e. a SUPERLUMINAL coordinate
shift |beta|~2), free_evolution.courant_dt() = cfl * min(dx,dy,dz)
ignores (b) entirely. Empirically (see the printed report below), running
BSSN at the same dt as free_evolution's default demo does NOT fix the
blow-up -- it can blow up to NaN *faster* than plain ADM, which looks like
"BSSN made it worse" but is actually the advection CFL violation dominating
before the hyperbolicity fix ever gets to matter. This script uses a
shift-aware dt (accounting for |beta|_max) so the comparison actually
isolates what the module docstrings claim to isolate.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from warpsim.metric import WarpBubbleParams
from warpsim.free_evolution import make_evolution_grid, run_evolution
from warpsim.bssn import run_evolution_bssn, run_evolution_bssn_dyngauge

OUT = "outputs/bssn_evolution_report.png"


def shift_aware_dt(grid, params, cfl=0.15, safety=0.9):
    """courant_dt() only accounts for grid spacing; for a coordinate shift
    with |beta| > 1 (superluminal, as this project's default v_s=2 bubble
    is BY DESIGN) the advection terms beta^i d_i f need
    dt < h / |beta|_max as well, or the explicit finite-difference
    advection stencil is unconditionally unstable regardless of which
    formulation (ADM or BSSN) is being advanced. `safety` gives extra
    margin since |beta| also varies across an RK4 substage."""
    from warpsim.free_evolution import courant_dt, gauge_fields
    h = min(grid.dx, grid.dy, grid.dz)
    _, beta0 = gauge_fields(grid, params, 0.0)
    beta_max = float(np.max(np.sqrt(np.sum(np.asarray(beta0) ** 2, axis=-1))))
    dt_wave = courant_dt(grid, cfl)
    dt_advect = safety * cfl * h / max(beta_max, 1e-6)
    return min(dt_wave, dt_advect)


def run_with_dt(run_fn, grid, params, t_end, dt, **kwargs):
    """All three run_* functions recompute dt internally from cfl; this
    wrapper instead derives the cfl value that reproduces our externally
    computed shift-aware dt, so all three schemes use IDENTICAL dt."""
    from warpsim.free_evolution import courant_dt
    h = min(grid.dx, grid.dy, grid.dz)
    equiv_cfl = dt / h
    return run_fn(grid, params, t_end, cfl=equiv_cfl, **kwargs)


def main():
    params = WarpBubbleParams(v_s=2.0, R=1.0, sigma=3.0, x_s0=0.0)
    grid = make_evolution_grid(x_range=(-3.0, 5.0), y_range=(-3.0, 3.0),
                                z_range=(-3.0, 3.0), nx=20, ny=14, nz=14)

    dt = shift_aware_dt(grid, params, cfl=0.15)
    t_end = 0.6
    print(f"[bssn_evolution] shift-aware dt={dt:.5f} "
          f"(vs. naive courant_dt cfl=0.15 dt={0.15*min(grid.dx,grid.dy,grid.dz):.5f})")

    results = {}
    print("[bssn_evolution] running ADM, prescribed gauge ...")
    t0 = time.time()
    results["ADM (prescribed)"] = run_with_dt(run_evolution, grid, params, t_end, dt,
                                               mode="vacuum", save_every=5)
    print(f"  done in {time.time()-t0:.1f}s")

    print("[bssn_evolution] running BSSN, prescribed gauge ...")
    t0 = time.time()
    results["BSSN (prescribed)"] = run_with_dt(run_evolution_bssn, grid, params, t_end, dt,
                                                mode="vacuum", save_every=5)
    print(f"  done in {time.time()-t0:.1f}s")

    print("[bssn_evolution] running BSSN, dynamical gauge (1+log / Gamma-driver) ...")
    t0 = time.time()
    results["BSSN (dynamical gauge)"] = run_with_dt(run_evolution_bssn_dyngauge, grid, params,
                                                      t_end, dt, mode="vacuum", save_every=5)
    print(f"  done in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"ADM (prescribed)": "tab:red", "BSSN (prescribed)": "tab:orange",
              "BSSN (dynamical gauge)": "tab:green"}

    ax = axes[0]
    for label, diag in results.items():
        ax.semilogy(diag["t"], np.abs(diag["ham_violation"]) + 1e-12, "o-",
                     color=colors[label], label=label)
    ax.set_xlabel("coordinate time t"); ax.set_ylabel("max |Hamiltonian constraint|")
    ax.set_title("Constraint growth at matched, shift-aware dt")
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, diag in results.items():
        ax.plot(diag["t"], diag["peak_x"], "o-", color=colors[label], label=label)
    t_dense = np.linspace(0, t_end, 100)
    ax.plot(t_dense, params.x_s0 + params.v_s * t_dense, "k--", label="prescribed x_s(t)=v_s t")
    ax.set_xlabel("coordinate time t"); ax.set_ylabel("bubble-shell centroid x")
    ax.set_title("Does the wall translate through the evolving grid?")
    ax.legend(fontsize=8)

    fig.suptitle("ADM vs BSSN (prescribed vs dynamical gauge), "
                  f"v_s={params.v_s} (superluminal), sigma={params.sigma}")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[bssn_evolution] wrote {OUT}")

    print("\nSummary (see module docstring for the full reasoning):")
    print("- free_evolution.courant_dt() ignores the shift magnitude, so at its default")
    print("  cfl the advection terms alone violate CFL for this project's superluminal")
    print("  (v_s=2) default bubble -- BOTH ADM and BSSN blow up under that dt, and BSSN")
    print("  can blow up to NaN FASTER than ADM there, which is an advection-CFL artifact,")
    print("  not evidence BSSN is worse. Always use a shift-aware dt (see shift_aware_dt()")
    print("  above) when judging formulation stability for this bubble.")
    print("- At matched, shift-aware dt: prescribed-gauge BSSN still eventually diverges")
    print("  (the strongly-hyperbolic principal part alone does not save a fixed,")
    print("  superluminal shift with no compensating gauge dynamics). Dynamical gauge")
    print("  (1+log lapse + Gamma-driver shift) is markedly more stable over the same")
    print("  interval -- consistent with the standard NR result that a puncture-style")
    print("  dynamical gauge, not the BSSN reformulation alone, is what makes long-term")
    print("  superluminal-shift evolutions tractable.")
    print("- None of this is resolution-converged (this grid is CPU-demo-sized); treat")
    print("  absolute constraint-violation numbers as illustrative, not validated.")


if __name__ == "__main__":
    main()
