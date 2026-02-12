"""Visualisation utilities for Omnidream TMS coil models and field maps.

Provides:
- 3-D coil wire-path plots (single coil or helmet array)
- Helmet layout scatter plot
- TI modulation depth and NTS V_peak field maps
- NTS firing schedule timing diagrams
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from custom_c_shaped_coil import CoilSpec, build_c_shaped_coil


PLOT_DIR = Path(__file__).resolve().parent / "visualizations"


def _ensure_plot_dir() -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


# ---------------------------------------------------------------------------
# Single-coil visualisation
# ---------------------------------------------------------------------------

def visualize_coil(spec: CoilSpec | None = None, save: bool = True) -> None:
    """Create 3-D visualisation of a single C-shaped coil."""
    if not HAS_MPL:
        print("matplotlib not available — skipping coil visualisation.")
        return

    if spec is None:
        spec = CoilSpec()
    coil, summary = build_c_shaped_coil(spec)

    # Extract wire points from all elements
    all_points = []
    for el in coil.elements:
        pts = el.get_points(np.eye(4), apply_deformation=False)
        all_points.append(pts)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for pts in all_points:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b-", linewidth=0.5)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("C-Shaped Miniature TMS Coil")

    if save:
        out = _ensure_plot_dir()
        plt.savefig(out / "coil_design_3d.png", dpi=200, bbox_inches="tight")
        # Additional views
        for i, (elev, azim) in enumerate([(0, 0), (90, 0), (0, 90)]):
            ax.view_init(elev, azim)
            plt.savefig(out / f"coil_design_view_{i}.png", dpi=200, bbox_inches="tight")
        print(f"Saved coil plots to {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Helmet layout
# ---------------------------------------------------------------------------

def visualize_helmet_layout(
    positions: np.ndarray,
    orientations: np.ndarray | None = None,
    save: bool = True,
    filename: str = "helmet_layout.png",
) -> None:
    """3-D scatter plot of coil positions on the helmet surface."""
    if not HAS_MPL:
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c="steelblue", s=60, edgecolors="k", linewidths=0.5,
    )

    if orientations is not None:
        scale = 5.0  # arrow length in mm
        ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            orientations[:, 0] * scale,
            orientations[:, 1] * scale,
            orientations[:, 2] * scale,
            color="red", alpha=0.6, arrow_length_ratio=0.2,
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Helmet Coil Layout ({len(positions)} coils)")

    if save:
        out = _ensure_plot_dir()
        plt.savefig(out / filename, dpi=200, bbox_inches="tight")
        print(f"Saved helmet layout → {out / filename}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# TI modulation depth map
# ---------------------------------------------------------------------------

def visualize_modulation_depth_map(
    M: np.ndarray,
    target_idx: int | None = None,
    save: bool = True,
    filename: str = "ti_modulation_depth.png",
) -> None:
    """1-D or 2-D plot of TI modulation depth across sample points."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(M, "b-", linewidth=0.5, alpha=0.7)
    if target_idx is not None:
        ax.axvline(target_idx, color="red", linestyle="--", label=f"Target (idx={target_idx})")
        ax.legend()
    ax.set_xlabel("Element index")
    ax.set_ylabel("Modulation depth M(r)")
    ax.set_title("TI Modulation Depth Across Brain Elements")

    if save:
        out = _ensure_plot_dir()
        plt.savefig(out / filename, dpi=200, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# NTS V_peak map
# ---------------------------------------------------------------------------

def visualize_v_peak_map(
    V_peak: np.ndarray,
    target_idx: int | None = None,
    save: bool = True,
    filename: str = "nts_v_peak.png",
) -> None:
    """Plot NTS V_peak across sample points."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(V_peak, "g-", linewidth=0.5, alpha=0.7)
    if target_idx is not None:
        ax.axvline(target_idx, color="red", linestyle="--", label=f"Target (idx={target_idx})")
        ax.legend()
    ax.set_xlabel("Element index")
    ax.set_ylabel("V_peak (a.u.)")
    ax.set_title("NTS Peak Membrane Potential Across Brain Elements")

    if save:
        out = _ensure_plot_dir()
        plt.savefig(out / filename, dpi=200, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# NTS firing schedule
# ---------------------------------------------------------------------------

def visualize_firing_schedule(
    fire_times: np.ndarray,
    amplitudes: np.ndarray,
    save: bool = True,
    filename: str = "nts_firing_schedule.png",
) -> None:
    """Timing diagram showing when each coil fires."""
    if not HAS_MPL:
        return

    n = len(fire_times)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.3)))

    order = np.argsort(fire_times)
    for rank, coil_idx in enumerate(order):
        t = fire_times[coil_idx] * 1e3  # convert to ms
        a = amplitudes[coil_idx]
        ax.barh(rank, width=0.1, left=t, height=0.6, color="steelblue", alpha=0.8)
        ax.text(t + 0.12, rank, f"C{coil_idx} (α={a:.2f})", va="center", fontsize=7)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing order")
    ax.set_title("NTS Firing Schedule")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i + 1}" for i in range(n)])

    if save:
        out = _ensure_plot_dir()
        plt.savefig(out / filename, dpi=200, bbox_inches="tight")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Creating coil visualisation...")
    visualize_coil()
    print("Done.")
