"""
visualize.py — Field Visualization System (project doc, section 17)

Scientific-diagnostic-style plots (labeled colorbars, physical units,
contour overlays) rather than decorative renders, per the project doc:
"The visualization is intended to be a scientific diagnostic tool, not
merely an artistic visualization."
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_field(X, Y, field, title, cbar_label, out_path, cmap="RdBu_r",
               symmetric=True, contour_bubble=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if symmetric:
        vmax = np.max(np.abs(field))
        vmax = vmax if vmax > 0 else 1.0
        vmin = -vmax
    else:
        vmin, vmax = np.min(field), np.max(field)

    im = ax.pcolormesh(X, Y, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if contour_bubble is not None:
        x_s, R = contour_bubble
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(x_s + R * np.cos(theta), R * np.sin(theta), "k--", lw=1,
                label=f"bubble wall (r_s = R = {R})")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_geodesic(traj_position, params, out_path, contour_bubble=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = traj_position[:, 1]
    y = traj_position[:, 2]
    ax.plot(x, y, color="crimson", lw=1.8, label="particle worldline (spatial track)")
    ax.scatter([x[0]], [y[0]], color="green", zorder=5, label="start")
    ax.scatter([x[-1]], [y[-1]], color="black", zorder=5, label="end")

    if contour_bubble is not None:
        x_s, R = contour_bubble
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(x_s + R * np.cos(theta), R * np.sin(theta), "k--", lw=1,
                label="bubble wall (t=0)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Geodesic trajectory (spatial projection)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_light_rays(rays, params, out_path, contour_bubble=None):
    """rays: list of position arrays (n_eval,4), each one light ray's
    spatial track. Plots them together to show lensing/deflection."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(rays)))
    for i, pos in enumerate(rays):
        x, y = pos[:, 1], pos[:, 2]
        ax.plot(x, y, color=colors[i], lw=1.5, label=f"ray {i+1} (y0={y[0]:.2f})")
        ax.scatter([x[0]], [y[0]], color=colors[i], marker="o", s=20, zorder=5)

    if contour_bubble is not None:
        x_s, R = contour_bubble
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(x_s + R * np.cos(theta), R * np.sin(theta), "k--", lw=1,
                label="bubble wall (t=0)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Null geodesics (light rays) through the warp geometry")
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sweep(x_values, y_values, xlabel, ylabel, title, out_path,
               y2_values=None, y2_label=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_values, y_values, "o-", color="crimson")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.set_title(title)

    if y2_values is not None:
        ax2 = ax.twinx()
        ax2.plot(x_values, y2_values, "s--", color="steelblue")
        ax2.set_ylabel(y2_label, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_normalization_error(tau, norm_err, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tau, norm_err, color="darkorange")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("proper time tau")
    ax.set_ylabel(r"$g_{ab}u^a u^b - (-1)$")
    ax.set_title("Geodesic four-velocity normalization drift (section 13)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
