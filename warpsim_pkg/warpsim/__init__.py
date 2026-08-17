"""
warpsim: Numerical General Relativity laboratory for warp-drive spacetime
geometries.

CRITICAL: JAX defaults to float32. For curvature computations (which involve
second derivatives of the metric, matrix inversion, and contractions of
rank-4 tensors) float32 catastrophically amplifies rounding error, especially
near the steep Alcubierre wall (large sigma). We force float64 (x64) globally
the moment this package is imported, before any other jax call happens
anywhere in the process.
"""
import jax
jax.config.update("jax_enable_x64", True)

__all__ = [
    "metric",
    "derivatives",
    "christoffel",
    "curvature",
    "stress_energy",
    "observer",
    "geodesic",
    "grid",
    "validation",
]
