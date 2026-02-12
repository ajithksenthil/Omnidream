#!/usr/bin/env python3
"""Generate plain-language README visuals for Omnidream.

These images are intentionally simplified so non-technical readers can
understand the system at a glance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#F8F7F3",
            "axes.facecolor": "#F8F7F3",
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "savefig.facecolor": "#F8F7F3",
        }
    )


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=2,
        edgecolor="#1F2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#111827", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        color="#374151",
    )
    ax.add_patch(arr)


def fig_overview():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "How Omnidream Works (Plain Language)",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.87,
        "From coil design to a safe stimulation plan",
        ha="center",
        va="center",
        fontsize=14,
        color="#374151",
    )

    xs = [0.05, 0.245, 0.44, 0.635, 0.83]
    labels = [
        "1) Build\nminiature coil",
        "2) Simulate fields\nin a head model",
        "3) Optimize settings\nfor target depth",
        "4) Enforce safety\nlimits at each step",
        "5) Output a time plan\n(alpha(t))",
    ]
    colors = ["#DCEBFA", "#E6F4EA", "#FDECC8", "#FADADD", "#E9E0F7"]
    w, h, y = 0.14, 0.26, 0.48
    for x, label, c in zip(xs, labels, colors):
        _box(ax, x, y, w, h, label, c)

    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w + 0.01, y + h / 2, xs[i + 1] - 0.01, y + h / 2)

    ax.text(
        0.5,
        0.22,
        "Result: A stimulation recipe that is focused, constrained by safety, and inspectable.",
        ha="center",
        va="center",
        fontsize=13,
        color="#1F2937",
    )

    path = OUT_DIR / "01_overview_pipeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_safety():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Safety Guardrails Explained", pad=18)

    metrics = ["SAR (W/kg)", "Temperature (C)", "Current (A)", "Voltage (V)"]
    limits = np.array([3.2, 41.0, 5.0, 60.0], dtype=float)
    example = np.array([2.5, 39.0, 3.8, 45.0], dtype=float)
    usage = example / limits
    y = np.arange(len(metrics))

    ax.barh(y, np.ones_like(y), color="#E5E7EB", edgecolor="none", height=0.52, label="hard limit")
    ax.barh(y, usage, color="#93C5FD", edgecolor="none", height=0.52, label="example run")

    for i, (u, ev, lv) in enumerate(zip(usage, example, limits)):
        ax.text(
            min(u + 0.02, 0.98),
            i,
            f"{ev:.2g}/{lv:.2g}",
            va="center",
            ha="left",
            color="#111827",
            fontsize=11,
        )

    ax.axvline(1.0, color="#991B1B", linewidth=2, linestyle="--")
    ax.text(1.005, -0.45, "limit", color="#991B1B", fontsize=11, ha="left", va="center")

    ax.set_xlim(0, 1.08)
    ax.set_yticks(y, metrics)
    ax.set_xlabel("Fraction of limit (1.0 means at the threshold)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    fig.text(
        0.5,
        0.02,
        "The optimizer is allowed to improve stimulation quality only while staying inside these limits.",
        ha="center",
        color="#1F2937",
        fontsize=12,
    )

    path = OUT_DIR / "02_safety_guardrails.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_worlds_to_path():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("From Many Candidate Worlds to One Executable Path", pad=18)

    n = 55
    x = rng.uniform(0.1, 0.95, n)
    y = 1.2 - 0.9 * x + rng.normal(0, 0.08, n)
    y = np.clip(y, 0.08, 1.1)

    ax.scatter(x, y, s=55, color="#9CA3AF", alpha=0.75, label="candidate worlds")

    idx = np.argsort(x)[:3].tolist() + [np.argmax(x)]
    path_pts = np.array([[x[i], y[i]] for i in idx])
    path_pts = path_pts[np.argsort(path_pts[:, 0])]

    ax.plot(path_pts[:, 0], path_pts[:, 1], color="#2563EB", linewidth=3, marker="o", label="chosen trajectory")
    for i, (px, py) in enumerate(path_pts):
        ax.text(px + 0.01, py + 0.015, f"W{i}", fontsize=10, color="#1E3A8A")

    ax.set_xlabel("Integration / coherence side (higher right)")
    ax.set_ylabel("Energy cost side (lower is easier)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.text(
        0.5,
        0.02,
        "Intuition: we start with many feasible states, then pick a safe low-barrier route between desired states.",
        ha="center",
        color="#1F2937",
        fontsize=12,
    )

    path = OUT_DIR / "03_worlds_to_trajectory.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_daily_view():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "What a Typical Day Looks Like with Omnidream",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#111827",
    )

    y = 0.55
    entries = [
        ("Morning", "Run calibration or synthetic smoke", "#DCEBFA"),
        ("Midday", "Review dashboards and safety margins", "#E6F4EA"),
        ("Afternoon", "Choose worlds and generate trajectory", "#FDECC8"),
        ("Evening", "Save atlas + share reproducible artifacts", "#E9E0F7"),
    ]

    x0, w, h, gap = 0.06, 0.2, 0.26, 0.045
    for i, (title, body, color) in enumerate(entries):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, f"{title}\n\n{body}", color)
        if i < len(entries) - 1:
            _arrow(ax, x + w + 0.008, y + h / 2, x + w + gap - 0.008, y + h / 2)

    ax.text(
        0.5,
        0.25,
        "You can always answer: What was run, what improved, and whether safety held.",
        ha="center",
        va="center",
        fontsize=13,
        color="#1F2937",
    )

    path = OUT_DIR / "04_daily_workflow.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    fig_overview()
    fig_safety()
    fig_worlds_to_path()
    fig_daily_view()
    print(f"Wrote images to: {OUT_DIR}")


if __name__ == "__main__":
    main()
