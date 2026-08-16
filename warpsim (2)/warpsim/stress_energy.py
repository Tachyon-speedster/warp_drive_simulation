"""
stress_energy.py — Stress-Energy Tensor (project doc, section 9)

MATH
----
Einstein's field equations (geometrized units, G = c = 1):

    G_{ab} = 8*pi*T_{ab}   =>   T_{ab} = G_{ab} / (8*pi)

IMPORTANT CONCEPTUAL POINT (kept from the project doc, section 9):
This project does NOT start from an assumed matter content and derive the
resulting geometry (the "forward" problem most GR textbooks teach first).
It does the INVERSE problem: the warp-bubble metric is *chosen* first
(metric.py), its Einstein tensor is computed from pure geometry
(curvature.py), and T_{ab} here is simply *read off* as "whatever
stress-energy distribution Einstein's equations would require to produce
this geometry." This is standard practice in the warp-drive literature
(Alcubierre 1994, Natario 2002, etc.) and is exactly how those papers
establish that exotic (negative-energy-density) matter is required — they
never assumed exotic matter, they *derived* the requirement from geometry.
"""
from __future__ import annotations
import jax.numpy as jnp

EIGHT_PI = 8.0 * jnp.pi


def stress_energy_tensor(Einstein):
    """T_{ab} = G_{ab} / (8 pi)."""
    return Einstein / EIGHT_PI
