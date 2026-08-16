#!/usr/bin/env bash
# Build the fd_engine pybind11 extension module in-place.
set -euo pipefail
cd "$(dirname "$0")"

PY_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -O3 -march=native -Wall -shared -std=c++17 -fPIC \
    -I"${PY_INCLUDE}" -I"${PYBIND_INCLUDE}" -I/usr/include/eigen3 \
    fd_engine.cpp -o "fd_engine${EXT_SUFFIX}"

echo "Built fd_engine${EXT_SUFFIX}"
