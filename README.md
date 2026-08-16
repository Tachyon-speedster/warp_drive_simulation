# Warp-Bubble Numerical GR Simulator

A computational laboratory for studying Alcubierre-type warp-drive
spacetime geometries: metric -> curvature -> implied stress-energy ->
observer-measured energy density -> energy conditions -> geodesics
(massive and null/light), all differentiated exactly via automatic
differentiation, with a legacy finite-difference engine kept for
validation, and a compiled C++ engine for fast grid-scale finite-
difference sweeps.

**This project makes no claim that a physical warp drive is buildable.**
It computes what General Relativity says the geometry, curvature, and
required stress-energy *would be* for a chosen bubble metric, and reports
where that requires negative energy density / violates energy conditions.
See `MATH.md` for the mathematics behind every formula.

## Setup

```bash
pip install jax jaxlib numpy scipy matplotlib pybind11 --break-system-packages
sudo apt-get install libeigen3-dev          # for the optional C++ engine
cd cpp && ./build.sh                        # builds fd_engine*.so
```

## Run the core demo (sections 1-19)

```bash
python3 run_demo.py
```

Runs the full validation suite, compares autodiff vs finite-difference
derivatives at the bubble wall, computes the curvature/stress-energy
pipeline + 3+1 (ADM) decomposition at the bubble center, generates 2D
field maps (Ricci scalar, `G_tt`, energy density), integrates a massive-
particle geodesic with normalization-drift tracking, and cross-checks a
grid against the compiled C++ engine. Figures land in `outputs/`.

## Run the milestones-2 demo (energy conditions, light rays, sweeps)

```bash
python3 run_milestones2.py
```

Produces (all in `outputs/`):
- `nec_map.png` — Null Energy Condition violation field (broader than the
  WEC-violation region in `energy_density_map.png`, as GR predicts NEC is
  the weaker/more-easily-violated condition).
- `light_ray_tracing.png` — null geodesics traced through the geometry.
  For the default superluminal (`v_s=2`) bubble, rays passing close to
  the wall show strong deflection — a real horizon-like lensing effect
  documented in `MATH.md`, not a numerical artifact (four-velocity
  normalization drift stays ~1e-10 through the whole integration).
- `sweep_velocity.png`, `sweep_radius.png`, `sweep_sigma.png` — integrated
  exotic-energy requirement and peak curvature vs. each bubble parameter.

## Run the milestones-3 demo (ADM constraints, 3D exotic mass, horizon threshold)

```bash
python3 run_milestones3.py
```

Produces:
- Console report: extrinsic curvature `K_ij` computed two independent ways
  (agree to machine precision), and ADM Hamiltonian/momentum constraint
  residuals at several points (~1e-14 — this is the strongest whole-
  pipeline correctness check in the project, since `K_ij` and `T_ab` come
  from two completely separate derivation chains).
- A true 3D volume integral of exotic energy (`integrated_negative_energy_3d`),
  vs. the faster 2D-slice proxy used for parameter sweeps.
- `deflection_vs_vs.png` — light deflection angle vs. bubble velocity,
  swept across the `v_s=1` subluminal/superluminal threshold.

## Project map

```
warpsim/
  metric.py             Alcubierre metric + bubble trajectory/shape params   [1]
  derivatives.py         autodiff (primary) + finite-difference (legacy)     [3,18]
  christoffel.py          inverse metric + Christoffel symbols               [2,4]
  curvature.py             Riemann/Ricci/Ricci scalar/Einstein tensor        [5-8]
  stress_energy.py          implied T_ab from the Einstein tensor            [9]
  observer.py                four-velocity, proper time, energy density      [10,11]
  adm.py                       3+1 decomposition + extrinsic curvature +     [14,21-23]
                                  Hamiltonian/momentum constraints
  geodesic.py                    massive + null geodesic integration         [12,13,+]
  energy_conditions.py             NEC / WEC / DEC checks                    [+]
  grid.py                            2D + 3D spatial grids, vmapped field    [15,16]
                                        evaluator
  parameter_sweep.py                   sweeps over v_s/R/sigma, 2D & 3D      [+]
                                          exotic-mass integrals
  visualize.py                           field/geodesic/ray/sweep plots      [17]
  validation.py                            full validation suite incl. ADM   [19]
                                              constraints
  cpp_bridge.py                              wrapper around the C++ engine
cpp/
  fd_engine.cpp        C++ (Eigen) finite-difference GR pipeline, grid-vectorized
  build.sh              compiles fd_engine*.so via pybind11
run_demo.py            core pipeline demo (sections 1-19)
run_milestones2.py     energy conditions, null geodesics/lensing, parameter sweeps
run_milestones3.py     ADM constraints, 3D exotic mass, v_s=1 threshold sweep
MATH.md                full mathematical reference / how to tweak the physics
```

Bracketed numbers refer to sections of the original project spec; `[+]`
marks milestones built beyond the original numbered sections.

## Known next milestones (not yet implemented)

Full dynamic (time-evolving) 3+1 numerical relativity — everything here
uses a *prescribed* metric at each instant, not an evolved one — and a
finer impact-parameter sweep to map the exact light-capture cross-section
near the wall (see the last section of `MATH.md`).
