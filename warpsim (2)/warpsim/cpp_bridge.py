"""
cpp_bridge.py — bridge to the compiled C++ legacy finite-difference engine
(cpp/fd_engine.cpp), used for full-grid FD-vs-autodiff comparisons and to
give the legacy path a real speedup over pure-Python nested loops, per the
user's request to "use C++ wherever required for speedups."

Import this lazily / with a clear error message: the C++ extension must be
built first via cpp/build.sh (see README.md).
"""
from __future__ import annotations
import os
import sys
import numpy as np

_CPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
if _CPP_DIR not in sys.path:
    sys.path.insert(0, _CPP_DIR)

try:
    import fd_engine as _fd_engine  # the compiled pybind11 module
    HAVE_CPP_ENGINE = True
except ImportError:
    _fd_engine = None
    HAVE_CPP_ENGINE = False


def evaluate_grid_fd_cpp(coords_flat, params, h_outer=1e-4, h_inner=1e-4):
    """coords_flat: (N,4) float64 numpy array [t,x,y,z].
    Returns dict of flat float64 arrays: R_scalar, G_tt, G_tx, T_tt,
    energy_density -- computed via the compiled C++ finite-difference
    pipeline (fd_engine.cpp), NOT the autodiff pipeline."""
    if not HAVE_CPP_ENGINE:
        raise ImportError(
            "fd_engine C++ extension is not built. Run cpp/build.sh first."
        )
    coords_flat = np.ascontiguousarray(coords_flat, dtype=np.float64)
    return _fd_engine.evaluate_grid(
        coords_flat, params.v_s, params.R, params.sigma, params.x_s0,
        h_outer, h_inner,
    )
