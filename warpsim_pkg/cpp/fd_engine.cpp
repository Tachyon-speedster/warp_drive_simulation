// fd_engine.cpp
//
// C++ speedup module for the LEGACY finite-difference derivative engine
// (project doc sections 3 & 18: "the old finite-difference engine should
// be retained for validation and comparison rather than immediately
// deleted"). The primary/precise pipeline is the JAX autodiff pipeline in
// Python (warpsim/derivatives.py, christoffel.py, curvature.py) — it is
// exact to float64 precision and already JIT+vmap compiled, so it does not
// need a C++ rewrite for correctness OR speed.
//
// What DOES benefit from C++: sweeping the two-level finite-difference
// engine (FD of FD -> Riemann) over a full 2D grid in Python is a tight
// nested-loop numerical kernel with no vectorization opportunity in
// NumPy/JAX (each grid point needs ~40 metric evaluations for the 2nd-
// derivative stencils). This module reimplements exactly that pipeline in
// C++ with Eigen for a large constant-factor speedup, and is used
// specifically to (a) validate the Python FD engine against an
// independent implementation and (b) produce the FD-vs-autodiff error map
// over a full grid fast enough to be interactive.
//
// Build: see build.sh in this directory.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cmath>
#include <array>

namespace py = pybind11;
using Mat4 = Eigen::Matrix<double, 4, 4>;
using Vec4 = Eigen::Matrix<double, 4, 1>;

struct Params {
    double v_s;
    double R;
    double sigma;
    double x_s0;
};

static inline double shape_function(double r_s, double R, double sigma) {
    double num = std::tanh(sigma * (r_s + R)) - std::tanh(sigma * (r_s - R));
    double den = 2.0 * std::tanh(sigma * R);
    return num / den;
}

static inline Mat4 metric_tensor(const Vec4 &coords, const Params &p) {
    double t = coords(0), x = coords(1), y = coords(2), z = coords(3);
    double x_s = p.x_s0 + p.v_s * t;
    double r_s = std::sqrt((x - x_s) * (x - x_s) + y * y + z * z);
    double f = shape_function(r_s, p.R, p.sigma);
    double v_s = p.v_s;

    Mat4 g = Mat4::Zero();
    g(0, 0) = -(1.0 - v_s * v_s * f * f);
    g(0, 1) = -v_s * f;
    g(1, 0) = -v_s * f;
    g(1, 1) = 1.0;
    g(2, 2) = 1.0;
    g(3, 3) = 1.0;
    return g;
}

// dg[a](m,n) = d g_{mn} / dx^a, central difference, O(h^2)
static inline std::array<Mat4, 4> metric_first_derivative(const Vec4 &coords,
                                                            const Params &p,
                                                            double h) {
    std::array<Mat4, 4> dg;
    for (int a = 0; a < 4; ++a) {
        Vec4 step = Vec4::Zero();
        step(a) = h;
        Mat4 gp = metric_tensor(coords + step, p);
        Mat4 gm = metric_tensor(coords - step, p);
        dg[a] = (gp - gm) / (2.0 * h);
    }
    return dg;
}

// Gamma[a](b,c) = Gamma^a_{bc}
static inline std::array<Mat4, 4> christoffel(const Mat4 &g, const Mat4 &g_inv,
                                               const std::array<Mat4, 4> &dg) {
    std::array<Mat4, 4> Gamma;
    for (int a = 0; a < 4; ++a) Gamma[a] = Mat4::Zero();

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                double sum = 0.0;
                for (int d = 0; d < 4; ++d) {
                    double term = dg[b](d, c) + dg[c](d, b) - dg[d](b, c);
                    sum += g_inv(a, d) * term;
                }
                Gamma[a](b, c) = 0.5 * sum;
            }
        }
    }
    return Gamma;
}

static inline std::array<Mat4, 4> christoffel_at(const Vec4 &coords,
                                                   const Params &p, double h,
                                                   Mat4 &g_out, Mat4 &g_inv_out) {
    Mat4 g = metric_tensor(coords, p);
    Mat4 g_inv = g.inverse();
    auto dg = metric_first_derivative(coords, p, h);
    g_out = g;
    g_inv_out = g_inv;
    return christoffel(g, g_inv, dg);
}

// Full FD-of-FD Riemann/Ricci/Einstein/energy-density pipeline at one point.
// h_outer: step for differentiating Gamma (outer FD pass)
// h_inner: step for differentiating g inside each Gamma evaluation (inner FD pass)
struct PointResult {
    double R_scalar;
    double G_tt;
    double G_tx;
    double T_tt;
    double energy_density;
};

static inline PointResult evaluate_point(const Vec4 &coords, const Params &p,
                                          double h_outer, double h_inner) {
    Mat4 g, g_inv;
    auto Gamma0 = christoffel_at(coords, p, h_inner, g, g_inv);

    // dGamma[d][a](b,c) = d Gamma^a_{bc} / dx^d, central difference
    std::array<std::array<Mat4, 4>, 4> dGamma; // [d][a]
    for (int d = 0; d < 4; ++d) {
        Vec4 step = Vec4::Zero();
        step(d) = h_outer;
        Mat4 gp_unused, ginv_unused;
        auto Gp = christoffel_at(coords + step, p, h_inner, gp_unused, ginv_unused);
        auto Gm = christoffel_at(coords - step, p, h_inner, gp_unused, ginv_unused);
        for (int a = 0; a < 4; ++a) {
            dGamma[d][a] = (Gp[a] - Gm[a]) / (2.0 * h_outer);
        }
    }

    // Riemann^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
    //                   + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}
    // We only need the Ricci contraction R_{bd} = R^a_{bad}, so contract
    // over 'a' == the free upper index while summing directly -- avoids
    // materializing the full rank-4 tensor.
    Mat4 Ricci = Mat4::Zero();
    for (int b = 0; b < 4; ++b) {
        for (int d = 0; d < 4; ++d) {
            double sum = 0.0;
            for (int a = 0; a < 4; ++a) {
                double term1 = dGamma[a][a](b, d);       // d_a Gamma^a_{bd} (c==a)
                double term2 = dGamma[d][a](b, a);       // d_d Gamma^a_{ba} (c==a in R^a_{bad}, with c=a,d=d... )
                double term3 = 0.0, term4 = 0.0;
                for (int e = 0; e < 4; ++e) {
                    term3 += Gamma0[a](a, e) * Gamma0[e](b, d);
                    term4 += Gamma0[a](d, e) * Gamma0[e](b, a);
                }
                sum += term1 - term2 + term3 - term4;
            }
            Ricci(b, d) = sum;
        }
    }

    double R_scalar = 0.0;
    for (int b = 0; b < 4; ++b)
        for (int d = 0; d < 4; ++d)
            R_scalar += g_inv(b, d) * Ricci(b, d);

    Mat4 Einstein = Ricci - 0.5 * g * R_scalar;
    Mat4 T = Einstein / (8.0 * M_PI);

    // Eulerian (ADM normal) observer: alpha=1 for this metric family, so
    // n^a = (1, v_s*f, 0, 0) = (1, -beta^x, 0, 0). This matches
    // observer.normalize_eulerian_observer() in the Python autodiff path,
    // so the two engines are compared using the SAME observer definition.
    double beta_x = g(0, 1); // beta_i = g_{0i}; gamma_ij = delta_ij here so beta^x = beta_x = g_tx
    Vec4 n;
    n << 1.0, -beta_x, 0.0, 0.0;
    double rho = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            rho += T(a, b) * n(a) * n(b);

    PointResult r;
    r.R_scalar = R_scalar;
    r.G_tt = Einstein(0, 0);
    r.G_tx = Einstein(0, 1);
    r.T_tt = T(0, 0);
    r.energy_density = rho;
    return r;
}

// Evaluate over a full 2D grid (flattened). Returns 5 flat arrays.
py::dict evaluate_grid(py::array_t<double> coords_flat, double v_s, double R,
                        double sigma, double x_s0, double h_outer,
                        double h_inner) {
    auto buf = coords_flat.request();
    if (buf.ndim != 2 || buf.shape[1] != 4)
        throw std::runtime_error("coords_flat must have shape (N,4)");
    size_t n = buf.shape[0];
    double *ptr = static_cast<double *>(buf.ptr);

    Params p{v_s, R, sigma, x_s0};

    py::array_t<double> out_R(n), out_Gtt(n), out_Gtx(n), out_Ttt(n), out_rho(n);
    auto rR = out_R.mutable_unchecked<1>();
    auto rGtt = out_Gtt.mutable_unchecked<1>();
    auto rGtx = out_Gtx.mutable_unchecked<1>();
    auto rTtt = out_Ttt.mutable_unchecked<1>();
    auto rrho = out_rho.mutable_unchecked<1>();

    for (size_t i = 0; i < n; ++i) {
        Vec4 coords;
        coords << ptr[4 * i + 0], ptr[4 * i + 1], ptr[4 * i + 2], ptr[4 * i + 3];
        PointResult res = evaluate_point(coords, p, h_outer, h_inner);
        rR(i) = res.R_scalar;
        rGtt(i) = res.G_tt;
        rGtx(i) = res.G_tx;
        rTtt(i) = res.T_tt;
        rrho(i) = res.energy_density;
    }

    py::dict result;
    result["R_scalar"] = out_R;
    result["G_tt"] = out_Gtt;
    result["G_tx"] = out_Gtx;
    result["T_tt"] = out_Ttt;
    result["energy_density"] = out_rho;
    return result;
}

PYBIND11_MODULE(fd_engine, m) {
    m.doc() = "C++ finite-difference GR pipeline (legacy engine, grid speedup)";
    m.def("evaluate_grid", &evaluate_grid,
          py::arg("coords_flat"), py::arg("v_s"), py::arg("R"),
          py::arg("sigma"), py::arg("x_s0"),
          py::arg("h_outer") = 1e-4, py::arg("h_inner") = 1e-4,
          "Evaluate FD-based curvature/energy-density fields over a flat (N,4) coords array.");
}
