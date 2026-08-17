# Mathematical Reference — Warp-Bubble Numerical GR Simulator

This document is the "why" behind every formula in `warpsim/`, so you can
modify the physics yourself without re-deriving GR from scratch. Read it
top to bottom once; after that use it as a lookup table (each section
title matches a source file).

Conventions used everywhere in the code:
- Signature: `(-,+,+,+)`.
- Units: geometrized, `G = c = 1`.
- Coordinate order: `x^mu = (t, x, y, z)`, indices `0,1,2,3`.
- Greek-letter tensor names in comments use plain ASCII: `Gamma`, `mu`,
  `nu`, etc.

---

## 1. The metric (`metric.py`)

The Alcubierre warp-bubble line element:

```
ds^2 = -dt^2 + (dx - v_s(t) f(r_s) dt)^2 + dy^2 + dz^2
```

`v_s(t)` = bubble coordinate velocity, `r_s = sqrt((x-x_s(t))^2+y^2+z^2)`
is distance from the bubble center, and `f` is the **shape function**:

```
f(r_s) = [tanh(sigma*(r_s+R)) - tanh(sigma*(r_s-R))] / [2*tanh(sigma*R)]
```

Why this specific shape function: it is a smoothed top-hat. `tanh` gives a
`C^infinity` transition (all derivatives exist and are bounded), which
matters because we differentiate this function *at least twice* downstream
(Riemann tensor needs second derivatives of the metric). A raw step
function would make curvature literally infinite (a delta function) at the
wall; `tanh` spreads that into a finite, well-resolved shell of width
`~1/sigma`.

**To change the bubble shape**: edit only `shape_function`. Any smooth,
compactly-transitioning function with `f(0)=1, f(inf)=0` works (e.g. a
Gaussian-difference, a smoothed error function, etc.) — nothing else in
the codebase references the specific tanh formula.

**To change the trajectory**: edit `WarpBubbleParams.bubble_center_x` /
`bubble_velocity`. Right now it's uniform velocity (`x_s = x_s0 + v_s*t`);
an accelerating bubble just needs `x_s(t)` to be any differentiable
function of `t` — autodiff handles the rest automatically, no other file
needs to change.

Expanding the line element into `g_{mu nu}` components (multiply out the
square):

```
g_tt = -(1 - v_s^2 f^2)
g_tx = g_xt = -v_s f
g_xx = g_yy = g_zz = 1
```

## 2. Inverse metric (`christoffel.inverse_metric`)

Plain `jnp.linalg.inv`. Validated against the identity `g @ g^-1 = I`
(`validation.check_metric_inverse_identity`) at every run.

## 3 & 18. Metric derivatives — why autodiff replaced finite differences
(`derivatives.py`)

We need `d g_{mn}/dx^a` (Christoffel symbols) and `d^2 g_{mn}/(dx^a dx^b)`
(Riemann tensor). Two ways to get a derivative numerically:

**Finite differences (legacy, `*_fd` functions):**
```
d g/dx ~ [g(x+h) - g(x-h)] / (2h)
```
Truncation error is `O(h^2)`; but round-off error grows as `O(1/h)`
(subtracting two nearly-equal floats loses precision). There's an optimal
`h` (usually ~1e-5 to 1e-6 for float64) balancing the two — go smaller and
round-off dominates, go bigger and truncation dominates. For the
**second** derivative needed by the Riemann tensor, the legacy path
finite-differences an *already finite-differenced* quantity (Christoffel
symbols, which themselves came from a first FD pass) — this **compounds**
both error sources, and is markedly worse right at the bubble wall where
`f` changes fastest (large `sigma` = large second derivatives = the finite
step `h` needs to resolve very fine structure while simultaneously not
being so small round-off swamps it — often no good `h` exists at all near
a steep wall).

**Automatic differentiation (primary, `*_autodiff` functions):**
`jax.jacfwd` builds the *exact analytic derivative* via the chain rule
through every `tanh`/`sqrt`/arithmetic op in `metric_tensor`, evaluated to
float64 machine precision (~1e-16 relative error, from floating point
representation only — no discretization error at all). Nesting
`jacfwd(jacfwd(...))` gives exact second derivatives the same way, with no
error compounding, which is precisely the case that breaks finite
differences. This is why the FD engine is kept only as a validation
cross-check (`validation.check_fd_convergence`,
`derivatives.compare_engines`), not as the production path.

Empirically (see `run_demo.py` output): at the bubble wall, FD-vs-autodiff
disagreement is `~2.7e-1` at `h=1e-2`, shrinks to `~2.7e-5` at `h=1e-4`
(≈h^2 scaling, confirming both are computing the same thing and FD is
converging correctly), and floors out around `~3e-9` at `h=1e-6` (round-off
floor). Autodiff has none of this trade-off.

## 4. Christoffel symbols (`christoffel.py`)

```
Gamma^a_{bc} = (1/2) g^{ad} ( d_b g_{dc} + d_c g_{db} - d_d g_{bc} )
```

This is the unique symmetric (`Gamma^a_{bc}=Gamma^a_{cb}`), metric-
compatible (`nabla_a g_{bc}=0`) connection — the Levi-Civita connection.
Symmetry in the lower two indices is a direct algebraic consequence of the
formula (swap `b<->c` and the bracket is unchanged) and is checked in
`validation.check_christoffel_symmetry`.

## 5. Riemann curvature tensor (`curvature.py`)

```
R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
            + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}
```

Intuition: parallel-transport a vector around an infinitesimal closed loop
spanned by coordinate directions `c` and `d`; the Riemann tensor is (up to
sign convention) how much the vector fails to return to itself. It
vanishes identically iff spacetime is flat (can be brought to Minkowski
form everywhere by a coordinate change) — this is exactly what
`validation.check_flat_limit` tests by setting `v_s=0`.

Implementation note: in the autodiff engine we don't separately compute
"Christoffel symbols, then differentiate them" as two disconnected steps —
we build one JAX function `gamma_of_coords(coords) -> Gamma` (which
internally already differentiates the metric once) and call
`jax.jacfwd` on *that whole function*. This differentiates the entire
chain (metric -> inverse -> first derivative -> Christoffel formula)
exactly, rather than us hand-deriving a separate second-derivative-of-
metric formula and risking an algebra mistake.

## 6 & 7. Ricci tensor and scalar

```
R_{bd} = R^a_{bad}          (contract 1st & 3rd index of Riemann)
R      = g^{bd} R_{bd}
```

Ricci is a "trace" of Riemann — it throws away information about how
curvature depends on the *orientation* of the loop (that's what's left in
the fully independent, "Weyl", part of Riemann) and keeps only the part
tied to volume distortion. This is exactly the piece Einstein's equations
couple directly to matter.

## 8. Einstein tensor

```
G_{ab} = R_{ab} - (1/2) g_{ab} R
```

Constructed so that `nabla^a G_{ab} = 0` identically (the contracted
Bianchi identity) — this is *why* it, rather than `R_{ab}` alone, is the
correct left-hand side of Einstein's equations: it guarantees local
conservation of whatever stress-energy it's set equal to.

## 9. Stress-energy tensor (`stress_energy.py`)

```
G_{ab} = 8*pi*T_{ab}   =>   T_{ab} = G_{ab} / (8*pi)
```

We are solving the **inverse problem**: geometry chosen first (section 1),
`T_{ab}` computed from it. This is standard in the warp-drive literature
and is precisely how one proves exotic matter is required — you don't
assume it, the geometry demands it.

## 10 & 11. Observers, four-velocity, proper time (`observer.py`)

A physical (massive) observer's four-velocity must satisfy the timelike
normalization `g_{ab}u^a u^b = -1`. Two observer constructions are
provided:

- **Coordinate-static** (`normalize_static_observer`): `u^a=(1/sqrt(-g_tt),0,0,0)`.
  Only valid where `g_tt<0`. For a **superluminal** bubble (`v_s>1`),
  `g_tt = -(1-v_s^2 f^2)` turns *positive* deep inside the bubble
  (`f->1`), meaning a grid-fixed worldline is genuinely spacelike there —
  this is real Alcubierre-metric physics, not a bug.
- **Eulerian / ADM-normal** (`normalize_eulerian_observer`, the default
  used for field maps): `n^a = (1/alpha)(1, -beta^1,-beta^2,-beta^3)`,
  built from the lapse/shift (section 14). This is timelike *by
  construction* (`g_ab n^a n^b = -1` always, whenever `alpha` is real and
  positive), so it's the safe default deep inside superluminal bubbles.

Proper time: `dtau = sqrt(-g_ab dx^a dx^b)` for any timelike displacement.

## 12 & 13. Geodesic equation and its conserved normalization
(`geodesic.py`)

```
d^2x^a/dtau^2 + Gamma^a_{bc} (dx^b/dtau)(dx^c/dtau) = 0
```

Free-fall trajectories extremize proper time. Converted to a first-order
system in `(x^a, u^a)` and integrated with `scipy.integrate.solve_ivp`
(adaptive RK45). A first integral of this equation is exactly
`g_ab u^a u^b = const` — i.e. **if you start with a correctly normalized
four-velocity, the geodesic equation itself guarantees it stays
normalized for all tau**. Tracking `g_ab u^a u^b - (-1)` along the
numerically-integrated trajectory therefore isn't testing the physics
model, it's testing the *numerical integrator's* accuracy — a clean,
independent sanity check with no free parameters to fudge.

## 14. 3+1 (ADM) decomposition (`adm.py`)

```
ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i+beta^i dt)(dx^j+beta^j dt)
```
Read off from a general `g_{ab}`:
```
gamma_ij = g_ij
beta_i   = g_{0i}         (beta^i = gamma^{ij} beta_j)
alpha^2  = beta_i beta^i - g_00
```
For the Alcubierre metric specifically: `alpha=1` and `gamma_ij=delta_ij`
identically — ALL of the geometry's non-triviality lives in the shift
`beta^x = -v_s f`. This is the defining structural feature of the
Alcubierre construction (and also why the Eulerian observer above reduces
to the simple `n^a=(1, v_s f, 0, 0)`).

## 15, 16, 17. Grid, field evaluator, visualization (`grid.py`, `visualize.py`)

No new math — this is "evaluate the section-1-through-11 pipeline at every
point of a 2D array, using `jax.vmap` instead of a Python loop so the
whole grid compiles to one XLA kernel." `evaluate_grid_fields` is
deliberately generic (`_single_point_fields` returns a dict of whatever
scalars you want) so adding a new field to visualize (e.g. a specific
Weyl-tensor component) means adding one line to that dict, not writing a
new grid loop.

## 19. Validation suite (`validation.py`)

Each check corresponds to an exact mathematical identity or a known
physical limit; see the file's docstring for the full list and the
formulas each one tests (metric/inverse identity, symmetries, flat and
asymptotic limits, FD-vs-autodiff convergence).

## 20 (continued). Energy conditions — implemented (`energy_conditions.py`)

- **WEC** (`check_wec`): `T_ab u^a u^b >= 0` for the given timelike `u^a`
  — this is just `observer.energy_density`.
- **NEC** (`check_nec`): `T_ab k^a k^b >= 0` for null `k^a`, sampled over
  `n_theta` directions via `k(theta) = u + cos(theta) e1 + sin(theta) e2`,
  where `{e1,e2}` are unit spacelike vectors orthogonal to `u` (and to
  each other), built by Gram-Schmidt on the coordinate basis
  (`orthonormal_spatial_frame`). NEC is *weaker* than WEC (implied by it
  via a limiting argument), so a NEC-violating region should always
  contain or exceed the WEC-violating region — verified in
  `run_milestones2.py` (31.5% NEC-violating vs 30.5% WEC-violating on the
  demo grid).
- **DEC** (`check_dec`): WEC, plus the energy-flux vector
  `q^a = -T^a_b u^b` must be non-spacelike (`g_ab q^a q^b <= 0`).

## 14/16/17 (continued). Null geodesics & light-ray tracing
(`geodesic.null_ray_direction`, same `integrate_geodesic` engine)

A null four-velocity `k^a = (1, n^x,n^y,n^z)` must satisfy
`g_ab k^a k^b = 0`. Fixing a spatial direction `e^i` (unit, in the flat
coordinate sense) and solving for the correct rescaling `s` such that
`n^i = s e^i` makes the vector null reduces to a quadratic in `s`:

```
A s^2 + B s + C = 0
A = g_ij e^i e^j,   B = 2 g_ti e^i,   C = g_tt
```

(`null_ray_direction` solves this and returns the `+` root). The same
`integrate_geodesic` function integrates both massive and null geodesics
— the geodesic *equation* doesn't care whether the affine parameter is
proper time or not; only the initial-condition normalization differs
(`-1` massive vs `0` null), which is why `integrate_geodesic` takes a
`norm_target` argument purely for the diagnostic drift trace.

**Observed result (v_s=2, superluminal):** light rays passing close to
the bubble wall show large deflection, and rays launched *away* from an
approaching superluminal bubble never catch up to it at all (nothing
outruns the bubble's own coordinate speed once `v_s` exceeds light's
local coordinate speed). Both are genuine, known features of superluminal
Alcubierre bubbles: Alcubierre's original 1994 paper notes a horizon-like
causal structure forms at the leading edge of the bubble once `v_s > 1` —
outgoing signals from inside/near the front cannot escape forward, exactly
analogous to a photon sphere/horizon in black-hole spacetimes. If you set
`v_s < 1` (subluminal) this structure disappears and lensing becomes
smooth/weak instead. Worth sweeping `v_s` across the `v_s=1` threshold
with the ray tracer if you want to see the transition directly.

## 21-23. Extrinsic curvature + ADM constraints — implemented (`adm.py`)

**Extrinsic curvature**, derived from first principles rather than quoted
from a particular textbook sign convention: the unit normal covector to a
`t=const` slice is `n_a = (-alpha, 0,0,0)`. Since `alpha=1` identically
for the Alcubierre metric, `n_a` is a CONSTANT covector field, so
`nabla_a n_b = d_a n_b - Gamma^c_{ab} n_c = Gamma^0_{ab}` (only the
`c=0` term of the sum survives, since `n_c` is zero except `n_0=-1`).
Restricting to spatial `a=i,b=j` (where the projector onto the slice is
trivial, since `n_i=0`):

```
K_ij = -nabla_i n_j = -Gamma^0_{ij}
```

Cross-checked against the independent textbook formula
`K_ij = (1/2)(d_i beta_j + d_j beta_i)` (valid because `alpha=1` and
`gamma_ij=delta_ij` is time-independent) — the two agree to machine
precision (`extrinsic_curvature` vs `extrinsic_curvature_from_shift`).

**ADM constraints** (Hamiltonian + momentum) are the "initial data"
consistency conditions any valid spacetime slicing must satisfy:

```
Hamiltonian:  (3)R + K^2 - K_ij K^ij = 16*pi*rho_ADM
Momentum:     D_j(K^ij - gamma^ij K) = 8*pi*j^i
```

With `(3)R=0` (flat spatial slices) and `D_j = partial_j` (flat spatial
connection) for this metric family, both reduce to closed-form
expressions built from `K_ij` (algebraic + one more autodiff derivative
pass) and `T_ab` (already computed via the Einstein tensor). **This is
the single strongest correctness check in the whole project**: `K_ij` is
built purely from Christoffel symbols, while `rho_ADM`/`j^i` come from a
completely separate chain (Riemann -> Ricci -> Einstein -> `T_ab`) — if
Riemann/Ricci/Einstein had an algebra bug anywhere, these constraints
would NOT hold to numerical precision. Measured residuals at several
test points (on the wall, inside the bubble, off-axis, far away): all
`~1e-14` or smaller (`run_milestones3.py`).

## 14 (true completion). 3D exotic-mass volume integral (`grid.make_grid_3d`,
`parameter_sweep.integrated_negative_energy_3d`)

The 2D-slice integral used for parameter sweeps (`integrated_negative_energy`)
is a fast proxy with units of energy-density*area, not a true mass. For an
actual integrated exotic-mass number, `make_grid_3d` builds a genuine 3D
`(x,y,z)` grid and `integrated_negative_energy_3d` sums
`rho * dx*dy*dz` over cells where `rho<0`. This costs more (a `nx*ny*nz`-point
vmap instead of `nx*ny`) but is the physically correct quantity; the 2D
number remains useful for fast trend-spotting across a parameter sweep
since the bubble's axial symmetry means the 2D and 3D numbers move
together as `v_s`/`R`/`sigma` change, even though their absolute values
differ.

## v_s=1 threshold — light deflection sweep (`run_milestones3.py`)

Deflection angle of a grazing light ray (`y0=0.6`, impact parameter just
inside the bubble radius `R=1`) was swept across `v_s in [0.3, 3.0]`,
crossing the subluminal/superluminal boundary at `v_s=1`. Result: the
deflection grows **smoothly and monotonically** through `v_s=1` at this
impact parameter — no discontinuity is visible in this sweep. This does
NOT contradict the strong, non-monotonic ray-capture behavior documented
above (section "14/16/17 continued") for near-grazing incidence right at
the leading wall edge; that effect appears to require an impact parameter
much closer to the wall boundary than `y0=0.6` sampled here. If you want
to map the exact capture cross-section / confirm a literal horizon,
narrow the impact-parameter sweep (e.g. `y0` in `[0.9, 1.1]` in small
steps) rather than the velocity sweep done here.

`integrated_negative_energy` does a 2D Riemann-sum integral of
`sum(rho[rho<0]) * dx * dy` over the `z=0` grid slice — a fast proxy for
"how much exotic matter, integrated, does this configuration need."
Sweeping `v_s`, `R`, and `sigma` independently (`sweep_velocity`,
`sweep_radius`, `sweep_wall_thickness`) shows exotic-energy requirement
and peak curvature both growing monotonically with all three parameters
(observed: roughly quadratic-looking growth with `v_s`, consistent with
the textbook `v_s^2` scaling of the Alcubierre energy requirement). This
integral is a 2D-slice quantity, not a true 3D volume integral — extend
`make_grid`/`evaluate_grid_fields` to a 3D `(x,y,z)` grid and integrate
`dx*dy*dz` if you need an actual total exotic-mass number; by the
bubble's axial symmetry about the direction of motion, the 2D-slice
trends already tell you how the true 3D integral would move with each
parameter.

## 21-23 (true completion). Full ADM dynamical evolution equation (`warpsim/adm_evolution.py`)

Everything above through section 23 builds the algebraic 3+1 (ADM) *split*
of the metric (`adm.py`: lapse, shift, spatial metric, extrinsic curvature)
and checks the two ADM *constraint* equations (Hamiltonian + momentum).
Constraints alone only certify that a single `t=const` slice is valid
initial data for numerical relativity — they say nothing about whether the
geometry evolves from one slice to the next the way Einstein's equations
actually require. That is the literal meaning of "3+1 spacetime": a
foliation into 3D space **evolving** through 1D time, not a stack of
independent snapshots.

`adm_evolution.py` implements the missing half: the full nonlinear ADM
evolution equation for the extrinsic curvature (MTW / Baumgarte-Shapiro
convention, matching this project's `K_ij = -Gamma^0_ij` sign convention),

```
d/dt K_ij = beta^k D_k K_ij + K_ik D_j beta^k + K_kj D_i beta^k
            - D_i D_j alpha
            + alpha * [ (3)R_ij + K K_ij - 2 K_ik gamma^kl K_lj ]
            - 8*pi*alpha * [ S_ij - (1/2) gamma_ij (S - rho) ]
```

built from first principles at every step:
- `(3)R_ij` — spatial Ricci tensor of `gamma_ij`, via the exact same
  Riemann-from-Christoffel contraction pattern as `curvature.py`, just
  restricted to the 3D spatial slice (not assumed to vanish, even though
  it numerically does for this metric family).
- `D_i D_j alpha` — covariant Hessian of the lapse using the spatial
  Christoffel symbols (general; not hard-coded to zero even though
  `alpha=1` for the Alcubierre family).
- `S_ij`, `S`, `rho` — matter fields measured by the Eulerian observer,
  built from the general tensor projector `gamma^a_b = delta^a_b + n^a n_b`
  applied to `T_ab` (not a metric-specific shortcut).

This is then cross-checked against the *independent* autodiff time
derivative of the closed-form `K_ij(t,x,y,z) = -Gamma^0_ij(t,x,y,z)`
already used in `adm.py`. Because the Alcubierre metric is an exact
solution for **all** `t` (not just `t=0`), the two sides come from
completely different derivation paths — one via the full nonlinear RHS,
the other via a single autodiff derivative straight through the metric —
and must agree to machine precision at every point if the whole pipeline
is self-consistent.

Result: residuals `~1e-13` to `~1e-17` (machine precision) at every test
point, including directly on the bubble wall, for superluminal (`v_s=5`,
`sigma=20`, steep wall) and subluminal (`v_s=0.5`, `sigma=3`, wide wall)
regimes alike. This is the strongest correctness test in the project —
stronger than the constraints alone — because it verifies the *dynamical*
Einstein equations, not just a snapshot's consistency. Wired into
`validation.py` as `check_adm_evolution_equation`, part of
`run_full_validation`.

## Free evolution & its instability (`free_evolution.py`, `run_free_evolution.py`)

Everything above -- including `adm_evolution.py`'s dynamical check -- still
only ever *reads off* the exact closed-form metric at different `t`. It
never asks Einstein's equations to *predict* `gamma_ij`, `K_ij` at a later
time from data at an earlier time without already knowing the answer.
That predictive step is what "time-evolving 3+1 spacetime" means in
numerical relativity, and it's what `free_evolution.py` adds.

**Method.** Exact Alcubierre ADM initial data (`gamma_ij=delta_ij`, `K_ij`,
`alpha=1`, `beta^i`) is taken from the existing autodiff pipeline at
`t=0`. From then on, `gamma_ij(x,y,z)` and `K_ij(x,y,z)` are discretized
on a finite 3D grid and genuinely evolved forward via the full nonlinear
ADM evolution equations (same RHS as `adm_evolution.py`, generalized to a
non-flat, finite-differenced spatial metric — `gamma_ij` is *not* forced
to stay `delta_ij`), using 2nd-order centered finite differences + 5-point
Kreiss-Oliger dissipation + explicit 4th-order Runge-Kutta (method of
lines). `alpha`/`beta^i` are re-evaluated exactly from the closed form at
every RK substage (they're treated as the drive's externally imposed
control inputs, not dynamical fields — see the module docstring for the
physical justification).

**Validation of the finite-difference engine itself**, independent of the
question below: `spatial_christoffel_ricci_fd` was checked against the
existing autodiff spatial-Ricci computation (`adm_evolution.spatial_ricci`)
on a smooth conformally-flat test metric (`gamma_ij = psi(x,y,z)^4
delta_ij`) at increasing resolution; the discrepancy shrinks by ~4x each
time the grid spacing halves, confirming genuine 2nd-order convergence
(and `Gamma3`/`Ricci3` both come out exactly `0.0` for `gamma_ij=delta_ij`
input, as they must).

**Physics result.** Two matter-sourcing modes are run: `"vacuum"`
(`T_ab=0` for `t>0`) and `"frozen_source"` (the `t=0` matter distribution
held fixed in the local Eulerian frame while the geometry evolves). In
both:

- The curvature/exotic-energy shell (tracked via the `K_ij K^ij`-weighted
  centroid, since the shell is a ring at `r_s~R`, not a single point —
  see `track_bubble_peak`) does translate in the +x direction as the
  geometry evolves, at roughly (not exactly) the prescribed `v_s` — i.e.
  the *dynamically evolved* spacetime, not just the closed-form metric,
  moves the bubble structure through the grid.
- The ADM Hamiltonian constraint violation and the deviation from the
  exact analytic solution both grow rapidly — within roughly one
  wall-crossing time — regardless of matter mode.

The second point is expected, not a bug: standard (unconstrained) ADM is
only *weakly* hyperbolic, and it is well documented in the numerical-
relativity literature that free ADM evolutions of generic data,
especially with a large, spatially-varying shift vector (exactly what a
warp-drive bubble's `beta^x = -v_s f(r_s)` is), develop exponentially
growing constraint-violating modes on a timescale set by the local
curvature/shift-gradient scale. This is part of why the field moved to
strongly/symmetric-hyperbolic reformulations (BSSN, generalized harmonic)
for any long-term stable numerical evolution. Concretely for a warp drive,
it also sharpens the "implementability" question this project exists to
quantify: sustaining the bubble is not just a matter of finding the right
initial exotic-matter distribution once — the *dynamics* actively fight
to relax it away, so a real drive would need continuous, actively
controlled re-sourcing (or a fundamentally different, non-ADM control
scheme) to stay on the intended trajectory, not merely to counteract a
static energy requirement.

**Known next milestones for this piece specifically:** a BSSN
(conformal-traceless) reformulation of the same physics for long-term
stable evolution; evolving `alpha`/`beta^i` dynamically too (1+log
slicing, Gamma-driver shift) instead of prescribing them, to separate
gauge dynamics from the matter-sourcing question studied here; and a real
evolution equation for the matter sector (an actual exotic-matter/field
model) instead of the `vacuum`/`frozen_source` approximations.

## BSSN reformulation (`warpsim/bssn.py`, `run_bssn_evolution.py`)

Implements the first "known next milestone" above: the conformal-
traceless (Baumgarte-Shapiro-Shibata-Nakamura) reformulation of the same
evolution equations, promoting the contracted conformal connection
functions `Gamma~^i` to an independently evolved field (the structural
change that turns ADM's weakly-hyperbolic principal symbol into a
strongly-hyperbolic one — see the module docstring in `bssn.py` for the
full derivation and its provenance/validation). Two variants are
provided: `run_evolution_bssn` (same prescribed closed-form
`alpha`/`beta^i` as `free_evolution.py`, isolating the effect of the
reformulation alone) and `run_evolution_bssn_dyngauge` (adds 1+log
slicing + 2nd-order Gamma-driver shift, the standard "moving puncture"
gauge choice, transcribed from the published NRPy+ source).

**Correctness check.** `conformal_ricci` (the load-bearing piece — the
Ricci tensor rewritten in terms of the evolved `Gamma~^i` field) is
cross-checked against an independent direct finite-difference Ricci
computation on a smooth, genuinely non-conformally-flat test metric, and
shown to converge at the expected O(h^2) rate (`validate_conformal_ricci`,
runnable via `python3 -m warpsim.bssn`). This passes.

**Empirical finding: an advection-CFL confound, not (only) a
hyperbolicity result.** Running `run_evolution_bssn` at the same `cfl`
`run_free_evolution.py` uses for plain ADM does **not** fix the blow-up
described above — at that `dt`, BSSN can hit `NaN` *faster* than plain
ADM. The reason is a bug/gap in `free_evolution.courant_dt()`, not the
BSSN formulation: it sets `dt = cfl * min(dx,dy,dz)`, accounting only for
the wave/light-crossing CFL condition, and never checks the *advection*
CFL condition `dt < h / |beta|_max` that the `beta^i d_i f` terms need.
This project's default bubble is **superluminal by construction**
(`v_s=2`), so `|beta|_max > 1` and that second condition is silently
violated by the existing default `cfl=0.15`, for *both* ADM and BSSN —
the FD advection stencil is unconditionally unstable there regardless of
which formulation is being advanced. `run_bssn_evolution.py` adds a
`shift_aware_dt()` helper that takes the tighter of the two conditions,
and re-runs the comparison at matched dt across all three schemes.

**Result at matched, shift-aware dt** (`run_bssn_evolution.py`, default
params, one representative run — not resolution-converged, treat as
illustrative):
- Plain ADM and prescribed-gauge BSSN both still diverge to `NaN` within
  the tested interval; prescribed-gauge BSSN does *not* reliably outlast
  plain ADM here. The strongly-hyperbolic principal part alone is not
  sufficient to stabilize a *fixed*, superluminal shift with no
  compensating gauge dynamics at this resolution.
- BSSN with the dynamical (1+log / Gamma-driver) gauge is markedly more
  stable over the same interval and tracks the prescribed bubble
  trajectory `x_s(t)=v_s t` more closely than either prescribed-gauge run.
  This matches the standard NR result that a puncture-style dynamical
  gauge — not the BSSN reformulation by itself — is usually what makes
  large/superluminal-shift evolutions numerically tractable in practice.

**Practical implication for this project:** before drawing any further
stability conclusions from `free_evolution.py` or `bssn.py`, either (a)
always compute `dt` with a shift-aware CFL condition like
`shift_aware_dt()`, or (b) patch `free_evolution.courant_dt()` itself to
take `max(1, |beta|_max)` into account — it currently silently
under-constrains `dt` for any `v_s>1` configuration, which is this
project's default. This is a numerics bug, not a physics finding, but it
was actively confounding formulation comparisons until diagnosed here.

**Still open / not yet validated:** convergence of the constraint-growth
rates themselves under grid refinement (all runs above are single-
resolution, CPU-demo-sized grids); the Gamma-driver's `eta` damping
parameter has not been tuned or scanned; and the `Gamma~^i` RHS's
matter-coupling and shift-Hessian terms are implemented from the
published equations but not independently re-derived/cross-checked (see
`bssn.py`'s module docstring, "WHAT IS AND ISN'T VALIDATED HERE").
