"""Aggregate SimNIBS calibration outputs into project progress visuals and reports.

This script scans a simulations root for `calibration_summary.json` files and generates:
- project_status.json
- project_status.md
- position_status.json
- position_status.md
- emax_vs_didt.png
- abs_error_vs_didt.png
- best_error_over_time.png
- runs_per_session.png
- runs_per_position.png
- best_error_by_position.png
- best_emax_by_position.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Build progress dashboard from calibration runs")
    p.add_argument("--sim-root", default=str(here / "simulations"), help="Simulation root directory")
    p.add_argument("--out-dir", default=str(here / "monitoring"), help="Output directory for reports/plots")
    p.add_argument("--goal-e-vpm", type=float, default=7.2, help="Fallback target E-field (V/m)")
    p.add_argument(
        "--goal-tolerance-frac",
        type=float,
        default=0.05,
        help="Goal tolerance as fractional error (default 5%%)",
    )
    p.add_argument(
        "--pipeline-dir",
        default=None,
        help="Pipeline output directory (default: auto-detect from Omnidream/pipeline_output)",
    )
    return p.parse_args(argv)


def position_value_to_str(value) -> str:
    if isinstance(value, (list, tuple)):
        vals = []
        for v in value:
            try:
                vals.append(f"{float(v):g}")
            except Exception:
                vals.append(str(v))
        return ",".join(vals)
    if value is None:
        return "unknown"
    return str(value)


def make_position_key(centre, pos_ydir) -> str:
    return f"{position_value_to_str(centre)} -> {position_value_to_str(pos_ydir)}"


def parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_run_dt(run_dir: str | None, fallback: datetime | None) -> datetime | None:
    if run_dir:
        base = Path(run_dir).name
        m = re.search(r"_(\d{8})_(\d{6})$", base)
        if m:
            raw = f"{m.group(1)}_{m.group(2)}"
            try:
                return datetime.strptime(raw, "%Y%m%d_%H%M%S")
            except Exception:
                pass
    return fallback


def load_summaries(sim_root: Path) -> list[dict]:
    summaries = []
    for path in sorted(sim_root.rglob("calibration_summary.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        obj["_summary_path"] = str(path)
        obj["_session_dir"] = str(path.parent)
        obj["_session_name"] = obj.get("session_name") or path.parent.name
        summaries.append(obj)
    return summaries


def collect_runs(summaries: list[dict], fallback_goal_e: float) -> list[dict]:
    rows: list[dict] = []
    for s in summaries:
        created = parse_iso_dt(s.get("created_at"))
        centre = s.get("centre")
        pos_ydir = s.get("pos_ydir")
        position_key = make_position_key(centre, pos_ydir)
        target_e = (
            s.get("targets", {}).get("target_e_vpm")
            if isinstance(s.get("targets"), dict)
            else None
        )
        if target_e is None:
            target_e = fallback_goal_e
        for r in s.get("rows", []):
            if not isinstance(r, dict):
                continue
            row = dict(r)
            row["_session_name"] = s["_session_name"]
            row["_session_dir"] = s["_session_dir"]
            row["_summary_path"] = s["_summary_path"]
            row["_summary_created_at"] = s.get("created_at")
            row["_target_e_vpm"] = target_e
            row["_run_dt"] = parse_run_dt(r.get("run_dir"), created)
            row["_centre"] = centre
            row["_pos_ydir"] = pos_ydir
            row["_position_key"] = position_key
            rows.append(row)
    return rows


def as_float(x) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def build_status(summaries: list[dict], runs: list[dict], fallback_goal_e: float, tol_frac: float) -> dict:
    valid = [r for r in runs if as_float(r.get("e_max")) is not None]
    for r in valid:
        e = as_float(r.get("e_max"))
        target = as_float(r.get("_target_e_vpm")) or fallback_goal_e
        r["_abs_error"] = abs(e - target)
        r["_rel_error"] = abs(e - target) / max(abs(target), 1e-12)

    best = min(valid, key=lambda x: x["_abs_error"]) if valid else None
    hits = [r for r in valid if r["_rel_error"] <= tol_frac]

    latest = None
    dated = [r for r in valid if r.get("_run_dt") is not None]
    if dated:
        latest = max(dated, key=lambda x: x["_run_dt"])

    session_counts = defaultdict(int)
    for r in runs:
        session_counts[r.get("_session_name", "unknown")] += 1

    status = {
        "generated_at": datetime.now().isoformat(),
        "sim_root_session_count": len(summaries),
        "total_runs_discovered": len(runs),
        "valid_runs_with_e_max": len(valid),
        "goal": {
            "target_e_vpm": fallback_goal_e,
            "tolerance_fraction": tol_frac,
            "tolerance_percent": tol_frac * 100.0,
        },
        "goal_hits_count": len(hits),
        "goal_hit_rate_percent": (100.0 * len(hits) / len(valid)) if valid else 0.0,
        "runs_per_session": dict(sorted(session_counts.items(), key=lambda kv: kv[0])),
        "best_run": None,
        "latest_run": None,
    }

    if best is not None:
        status["best_run"] = {
            "session": best.get("_session_name"),
            "didt_as": as_float(best.get("didt_as")),
            "e_max_vpm": as_float(best.get("e_max")),
            "target_e_vpm": as_float(best.get("_target_e_vpm")) or fallback_goal_e,
            "abs_error_vpm": best.get("_abs_error"),
            "rel_error_percent": best.get("_rel_error", 0.0) * 100.0,
            "run_dir": best.get("run_dir"),
            "summary_path": best.get("_summary_path"),
        }

    if latest is not None:
        status["latest_run"] = {
            "session": latest.get("_session_name"),
            "run_timestamp": latest["_run_dt"].isoformat(),
            "didt_as": as_float(latest.get("didt_as")),
            "e_max_vpm": as_float(latest.get("e_max")),
            "run_dir": latest.get("run_dir"),
            "summary_path": latest.get("_summary_path"),
        }

    return status


def build_position_status(runs: list[dict], fallback_goal_e: float, tol_frac: float) -> dict:
    valid = [r for r in runs if as_float(r.get("e_max")) is not None]
    for r in valid:
        if "_abs_error" not in r or "_rel_error" not in r:
            e = as_float(r.get("e_max"))
            target = as_float(r.get("_target_e_vpm")) or fallback_goal_e
            r["_abs_error"] = abs(e - target)
            r["_rel_error"] = abs(e - target) / max(abs(target), 1e-12)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        grouped[r.get("_position_key", "unknown")] += [r]

    best_by_position = []
    runs_per_position = {}
    for pos_key, items in grouped.items():
        runs_per_position[pos_key] = len(items)
        best = min(items, key=lambda x: x["_abs_error"])
        latest = None
        dated = [x for x in items if x.get("_run_dt") is not None]
        if dated:
            latest = max(dated, key=lambda x: x["_run_dt"])

        best_by_position.append(
            {
                "position_key": pos_key,
                "centre": best.get("_centre"),
                "pos_ydir": best.get("_pos_ydir"),
                "run_count": len(items),
                "best_didt_as": as_float(best.get("didt_as")),
                "best_e_max_vpm": as_float(best.get("e_max")),
                "target_e_vpm": as_float(best.get("_target_e_vpm")) or fallback_goal_e,
                "best_abs_error_vpm": best.get("_abs_error"),
                "best_rel_error_percent": best.get("_rel_error", 0.0) * 100.0,
                "best_run_dir": best.get("run_dir"),
                "latest_run_timestamp": latest["_run_dt"].isoformat() if latest else None,
            }
        )

    best_by_position.sort(key=lambda x: x["best_abs_error_vpm"])
    goal_hits = [x for x in best_by_position if (x["best_rel_error_percent"] / 100.0) <= tol_frac]

    position_status = {
        "generated_at": datetime.now().isoformat(),
        "position_count": len(grouped),
        "runs_per_position": dict(sorted(runs_per_position.items(), key=lambda kv: kv[0])),
        "best_by_position": best_by_position,
        "goal_hits_position_count": len(goal_hits),
        "goal_hits_position_rate_percent": (
            100.0 * len(goal_hits) / len(best_by_position) if best_by_position else 0.0
        ),
    }
    return position_status


def write_markdown(path: Path, status: dict) -> None:
    lines = []
    lines.append("# Project Progress Dashboard")
    lines.append("")
    lines.append(f"- Generated: `{status['generated_at']}`")
    lines.append(f"- Sessions discovered: `{status['sim_root_session_count']}`")
    lines.append(f"- Runs discovered: `{status['total_runs_discovered']}`")
    lines.append(f"- Valid runs: `{status['valid_runs_with_e_max']}`")
    lines.append(
        f"- Goal: `E_max = {status['goal']['target_e_vpm']} V/m` "
        f"(tolerance `{status['goal']['tolerance_percent']:.2f}%`)"
    )
    lines.append(f"- Goal hit rate: `{status['goal_hit_rate_percent']:.2f}%`")
    lines.append("")

    best = status.get("best_run")
    if best:
        lines.append("## Best Run")
        lines.append(
            f"- Session `{best['session']}` | dI/dt `{best['didt_as']:.6g}` A/s | "
            f"E_max `{best['e_max_vpm']:.6g}` V/m"
        )
        lines.append(
            f"- Error `{best['abs_error_vpm']:.6g}` V/m "
            f"(`{best['rel_error_percent']:.3f}%`)"
        )
        lines.append(f"- Run dir: `{best['run_dir']}`")
        lines.append("")

    latest = status.get("latest_run")
    if latest:
        lines.append("## Latest Run")
        lines.append(
            f"- `{latest['run_timestamp']}` | Session `{latest['session']}` | "
            f"dI/dt `{latest['didt_as']:.6g}` A/s | E_max `{latest['e_max_vpm']:.6g}` V/m"
        )
        lines.append(f"- Run dir: `{latest['run_dir']}`")
        lines.append("")

    lines.append("## Runs Per Session")
    for k, v in status.get("runs_per_session", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Plots")
    lines.append("- `emax_vs_didt.png`")
    lines.append("- `abs_error_vs_didt.png`")
    lines.append("- `best_error_over_time.png`")
    lines.append("- `runs_per_session.png`")
    lines.append("- `runs_per_position.png`")
    lines.append("- `best_error_by_position.png`")
    lines.append("- `best_emax_by_position.png`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_position_markdown(path: Path, position_status: dict, limit: int = 25) -> None:
    lines = []
    lines.append("# Position-Level Progress Dashboard")
    lines.append("")
    lines.append(f"- Generated: `{position_status['generated_at']}`")
    lines.append(f"- Positions discovered: `{position_status['position_count']}`")
    lines.append(
        f"- Positions meeting tolerance: `{position_status['goal_hits_position_count']}` "
        f"(`{position_status['goal_hits_position_rate_percent']:.2f}%`)"
    )
    lines.append("")
    lines.append("## Best By Position")
    for item in position_status.get("best_by_position", [])[:limit]:
        lines.append(
            f"- `{item['position_key']}` | runs `{item['run_count']}` | "
            f"best dI/dt `{item['best_didt_as']:.6g}` A/s | "
            f"E_max `{item['best_e_max_vpm']:.6g}` V/m | "
            f"error `{item['best_abs_error_vpm']:.6g}` V/m ({item['best_rel_error_percent']:.3f}%)"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("- `runs_per_position.png`")
    lines.append("- `best_error_by_position.png`")
    lines.append("- `best_emax_by_position.png`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_plots(out_dir: Path, runs: list[dict], fallback_goal_e: float) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    valid = [r for r in runs if as_float(r.get("e_max")) is not None and as_float(r.get("didt_as")) is not None]
    if not valid:
        return []

    sessions = sorted(set(r.get("_session_name", "unknown") for r in valid))
    colors = {}
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(sessions):
        colors[s] = cmap(i % 10)

    # Plot 1: Emax vs dIdt
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in sessions:
        pts = [r for r in valid if r.get("_session_name") == s]
        pts.sort(key=lambda x: as_float(x.get("didt_as")))
        xs = [as_float(p.get("didt_as")) for p in pts]
        ys = [as_float(p.get("e_max")) for p in pts]
        ax.plot(xs, ys, marker="o", label=s, color=colors[s], linewidth=1.2)
    ax.axhline(fallback_goal_e, linestyle="--", color="black", linewidth=1.0, label=f"goal {fallback_goal_e} V/m")
    ax.set_xscale("log")
    ax.set_xlabel("dI/dt (A/s)")
    ax.set_ylabel("E_max (V/m)")
    ax.set_title("E_max vs dI/dt")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    p1 = out_dir / "emax_vs_didt.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Plot 2: absolute error vs dIdt
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in sessions:
        pts = [r for r in valid if r.get("_session_name") == s]
        pts.sort(key=lambda x: as_float(x.get("didt_as")))
        xs = [as_float(p.get("didt_as")) for p in pts]
        ys = []
        for p in pts:
            target = as_float(p.get("_target_e_vpm")) or fallback_goal_e
            ys.append(abs(as_float(p.get("e_max")) - target))
        ax.plot(xs, ys, marker="o", label=s, color=colors[s], linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dI/dt (A/s)")
    ax.set_ylabel("|E_max - target| (V/m)")
    ax.set_title("Absolute Target Error vs dI/dt")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    p2 = out_dir / "abs_error_vs_didt.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # Plot 3: best error over time
    dated = [r for r in valid if r.get("_run_dt") is not None]
    if dated:
        dated.sort(key=lambda x: x["_run_dt"])
        best_so_far = []
        cur = math.inf
        for r in dated:
            target = as_float(r.get("_target_e_vpm")) or fallback_goal_e
            err = abs(as_float(r.get("e_max")) - target)
            cur = min(cur, err)
            best_so_far.append(cur)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([r["_run_dt"] for r in dated], best_so_far, marker="o", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Run time")
        ax.set_ylabel("Best |E_max - target| so far (V/m)")
        ax.set_title("Convergence of Best Error Over Time")
        ax.grid(True, alpha=0.25)
        p3 = out_dir / "best_error_over_time.png"
        fig.tight_layout()
        fig.savefig(p3, dpi=160)
        plt.close(fig)
    else:
        p3 = None

    # Plot 4: runs per session
    counts = defaultdict(int)
    for r in valid:
        counts[r.get("_session_name", "unknown")] += 1
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(sorted(counts.keys()))
    vals = [counts[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_xlabel("Session")
    ax.set_ylabel("Run count")
    ax.set_title("Run Count per Session")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    p4 = out_dir / "runs_per_session.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    outs = [str(p1), str(p2), str(p4)]
    if p3 is not None:
        outs.append(str(p3))
    return outs


def build_position_plots(
    out_dir: Path, runs: list[dict], fallback_goal_e: float, top_k: int = 30
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    valid = [r for r in runs if as_float(r.get("e_max")) is not None and r.get("_position_key")]
    if not valid:
        return []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        grouped[r["_position_key"]].append(r)

    rows = []
    for pos, items in grouped.items():
        for r in items:
            if "_abs_error" not in r:
                target = as_float(r.get("_target_e_vpm")) or fallback_goal_e
                r["_abs_error"] = abs(as_float(r.get("e_max")) - target)
        best = min(items, key=lambda x: x["_abs_error"])
        rows.append(
            {
                "position": pos,
                "run_count": len(items),
                "best_e": as_float(best.get("e_max")),
                "best_abs_error": best.get("_abs_error"),
            }
        )

    # Order positions by best absolute error (closest to goal first)
    rows.sort(key=lambda x: x["best_abs_error"])
    rows_view = rows[:top_k]
    labels = [r["position"] for r in rows_view]

    # Plot A: runs per position
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, [r["run_count"] for r in rows_view])
    ax.set_xlabel("Position (centre -> pos_ydir)")
    ax.set_ylabel("Run count")
    ax.set_title("Runs per Position (Top by Goal Proximity)")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    p1 = out_dir / "runs_per_position.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Plot B: best absolute error by position
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, [r["best_abs_error"] for r in rows_view])
    has_positive_error = any((r["best_abs_error"] or 0.0) > 0.0 for r in rows_view)
    if has_positive_error:
        ax.set_yscale("log")
    ax.set_xlabel("Position (centre -> pos_ydir)")
    ax.set_ylabel("Best |E_max - target| (V/m)")
    ax.set_title("Best Goal Error by Position")
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    p2 = out_dir / "best_error_by_position.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # Plot C: best Emax by position + goal line
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, [r["best_e"] for r in rows_view], label="best E_max")
    ax.axhline(fallback_goal_e, linestyle="--", color="black", linewidth=1.0, label=f"goal {fallback_goal_e} V/m")
    ax.set_xlabel("Position (centre -> pos_ydir)")
    ax.set_ylabel("Best E_max (V/m)")
    ax.set_title("Best E_max by Position")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    p3 = out_dir / "best_emax_by_position.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    return [str(p1), str(p2), str(p3)]


# =====================================================================
# Pipeline / TI / NTS / Hybrid dashboard
# =====================================================================

def load_pipeline_summary(pipeline_dir: Path) -> dict | None:
    """Load the pipeline summary JSON if available."""
    summary_path = pipeline_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_ga_result(pipeline_dir: Path) -> dict | None:
    """Load the GA best-individual result."""
    ga_path = pipeline_dir / "ga_best.json"
    if not ga_path.exists():
        return None
    try:
        return json.loads(ga_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_basis_data(pipeline_dir: Path) -> dict | None:
    """Load basis field metadata from npz."""
    import numpy as np
    basis_path = pipeline_dir / "basis_data.npz"
    if not basis_path.exists():
        return None
    try:
        data = np.load(str(basis_path))
        return {
            "n_points": data["basis_matrix"].shape[0],
            "n_coils": data["basis_matrix"].shape[1],
            "target_idx": int(data["target_idx"][0]) if "target_idx" in data else None,
            "n_surface": len(data["surface_indices"]) if "surface_indices" in data else None,
        }
    except Exception:
        return None


def compute_pipeline_metrics(pipeline_dir: Path) -> dict:
    """Compute TI/NTS/hybrid metrics from pipeline output."""
    import numpy as np

    metrics: dict = {"mode": None}

    summary = load_pipeline_summary(pipeline_dir)
    ga_result = load_ga_result(pipeline_dir)
    basis_info = load_basis_data(pipeline_dir)

    if summary:
        metrics["mode"] = summary.get("mode")
        metrics["stages_run"] = summary.get("stages_run")
        metrics["elapsed_s"] = summary.get("elapsed_s")
        metrics["ga_best_fitness"] = summary.get("ga_best_fitness")
        metrics["max_coupling_k"] = summary.get("max_coupling_k")
        metrics["synthetic"] = summary.get("synthetic", False)

    if basis_info:
        metrics["basis_n_points"] = basis_info["n_points"]
        metrics["basis_n_coils"] = basis_info["n_coils"]
        metrics["basis_target_idx"] = basis_info["target_idx"]
        metrics["basis_n_surface"] = basis_info["n_surface"]

    if ga_result:
        metrics["ga_mode"] = ga_result.get("mode")
        metrics["ga_fitness"] = ga_result.get("fitness")
        metrics["ga_freq_carrier"] = ga_result.get("freq_carrier")
        metrics["ga_delta_freq"] = ga_result.get("delta_freq")
        metrics["ga_n_coils"] = len(ga_result.get("amplitudes", []))
        metrics["ga_amplitudes"] = ga_result.get("amplitudes")
        metrics["ga_group"] = ga_result.get("group")
        metrics["ga_fire_times"] = ga_result.get("fire_times")

        amplitudes = ga_result.get("amplitudes", [])
        group = ga_result.get("group", [])
        if amplitudes:
            metrics["mean_amplitude"] = float(np.mean(amplitudes))
            metrics["max_amplitude"] = float(np.max(amplitudes))
        if group:
            n_group0 = sum(1 for g in group if g < 0.5)
            n_group1 = sum(1 for g in group if g >= 0.5)
            metrics["ti_group_split"] = f"{n_group0}/{n_group1}"

        # Recompute TI / NTS metrics from basis + GA result
        basis_path = pipeline_dir / "basis_data.npz"
        if basis_path.exists():
            try:
                data = np.load(str(basis_path))
                basis_matrix = data["basis_matrix"]
                target_idx = int(data["target_idx"][0])
                surface_indices = data["surface_indices"]

                amps = np.array(amplitudes)
                grp = np.array(group)

                mode = ga_result.get("mode", "")
                if mode in ("TI", "hybrid"):
                    from ti_fields import compute_group_amplitudes, compute_modulation_depth
                    freq1 = ga_result.get("freq_carrier", 1000.0)
                    freq2 = freq1 + ga_result.get("delta_freq", 40.0)
                    A1, A2 = compute_group_amplitudes(amps, grp, basis_matrix, freq1, freq2)
                    M = compute_modulation_depth(A1, A2)
                    metrics["ti_M_target"] = float(M[target_idx])
                    metrics["ti_M_surface_max"] = float(np.max(M[surface_indices]))
                    metrics["ti_M_surface_mean"] = float(np.mean(M[surface_indices]))
                    if metrics["ti_M_surface_max"] > 0:
                        metrics["ti_selectivity"] = float(M[target_idx] / metrics["ti_M_surface_max"])
                    else:
                        metrics["ti_selectivity"] = float("inf")

                if mode in ("NTS", "hybrid"):
                    from nts_timing import compute_v_peak, compute_per_pulse_surface_max
                    from config import paper_baseline
                    nts_cfg = paper_baseline().nts
                    fire_times = np.array(ga_result.get("fire_times", np.zeros(len(amps))))
                    V_peak = compute_v_peak(amps, fire_times, basis_matrix,
                                            tau_m=nts_cfg.tau_m_s, q_pulse=nts_cfg.q_pulse)
                    metrics["nts_V_target"] = float(V_peak[target_idx])
                    metrics["nts_V_surface_max"] = float(np.max(V_peak[surface_indices]))
                    metrics["nts_V_surface_mean"] = float(np.mean(V_peak[surface_indices]))
                    per_pulse = compute_per_pulse_surface_max(amps, basis_matrix, surface_indices)
                    metrics["nts_per_pulse_max"] = float(np.max(per_pulse))
                    if metrics["nts_V_surface_max"] > 0:
                        metrics["nts_selectivity"] = float(V_peak[target_idx] / metrics["nts_V_surface_max"])
                    else:
                        metrics["nts_selectivity"] = float("inf")
            except Exception as e:
                metrics["metric_compute_error"] = str(e)

    return metrics


def write_pipeline_markdown(path: Path, metrics: dict) -> None:
    """Write pipeline-specific dashboard markdown."""
    lines = []
    lines.append("# Pipeline Results Dashboard")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Mode: `{metrics.get('mode', 'unknown')}`")
    lines.append(f"- Synthetic: `{metrics.get('synthetic', False)}`")
    if metrics.get("elapsed_s") is not None:
        lines.append(f"- Elapsed: `{metrics['elapsed_s']:.2f}s`")
    lines.append("")

    # Basis field info
    if metrics.get("basis_n_points"):
        lines.append("## Basis Fields")
        lines.append(f"- Sample points: `{metrics['basis_n_points']}`")
        lines.append(f"- Coil count: `{metrics['basis_n_coils']}`")
        lines.append(f"- Target index: `{metrics['basis_target_idx']}`")
        lines.append(f"- Surface elements: `{metrics['basis_n_surface']}`")
        lines.append("")

    # Coupling
    if metrics.get("max_coupling_k") is not None:
        lines.append("## Coupling")
        lines.append(f"- Max coupling coefficient k: `{metrics['max_coupling_k']:.6f}`")
        lines.append("")

    # GA results
    lines.append("## GA Optimization")
    lines.append(f"- Best fitness: `{metrics.get('ga_fitness', 'N/A')}`")
    lines.append(f"- Number of coils: `{metrics.get('ga_n_coils', 'N/A')}`")
    if metrics.get("mean_amplitude") is not None:
        lines.append(f"- Mean amplitude: `{metrics['mean_amplitude']:.4f}`")
        lines.append(f"- Max amplitude: `{metrics['max_amplitude']:.4f}`")
    if metrics.get("ti_group_split"):
        lines.append(f"- TI group split: `{metrics['ti_group_split']}`")
    if metrics.get("ga_freq_carrier"):
        lines.append(f"- Carrier frequency: `{metrics['ga_freq_carrier']:.1f} Hz`")
    if metrics.get("ga_delta_freq"):
        lines.append(f"- Beat frequency (Δf): `{metrics['ga_delta_freq']:.1f} Hz`")
    lines.append("")

    # TI metrics
    if metrics.get("ti_M_target") is not None:
        lines.append("## TI Metrics")
        lines.append(f"- Modulation depth at target: `{metrics['ti_M_target']:.6f}`")
        lines.append(f"- Modulation depth surface max: `{metrics['ti_M_surface_max']:.6f}`")
        lines.append(f"- Modulation depth surface mean: `{metrics['ti_M_surface_mean']:.6f}`")
        lines.append(f"- Selectivity ratio (M_target / M_surface_max): `{metrics['ti_selectivity']:.4f}`")
        lines.append("")

    # NTS metrics
    if metrics.get("nts_V_target") is not None:
        lines.append("## NTS Metrics")
        lines.append(f"- V_peak at target: `{metrics['nts_V_target']:.6f}`")
        lines.append(f"- V_peak surface max: `{metrics['nts_V_surface_max']:.6f}`")
        lines.append(f"- V_peak surface mean: `{metrics['nts_V_surface_mean']:.6f}`")
        lines.append(f"- Selectivity ratio (V_target / V_surface_max): `{metrics['nts_selectivity']:.4f}`")
        lines.append(f"- Per-pulse max surface exposure: `{metrics['nts_per_pulse_max']:.6f}`")
        lines.append("")

    if metrics.get("metric_compute_error"):
        lines.append(f"**Warning:** metric computation error: `{metrics['metric_compute_error']}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_pipeline_plots(out_dir: Path, pipeline_dir: Path) -> list[str]:
    """Generate TI/NTS-specific plots from pipeline output."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_paths: list[str] = []

    basis_path = pipeline_dir / "basis_data.npz"
    ga_path = pipeline_dir / "ga_best.json"
    if not basis_path.exists() or not ga_path.exists():
        return plot_paths

    try:
        data = np.load(str(basis_path))
        basis_matrix = data["basis_matrix"]
        target_idx = int(data["target_idx"][0])
        surface_indices = data["surface_indices"]

        ga_result = json.loads(ga_path.read_text(encoding="utf-8"))
        amps = np.array(ga_result["amplitudes"])
        group = np.array(ga_result["group"])
        mode = ga_result.get("mode", "")
    except Exception:
        return plot_paths

    # Plot TI modulation depth
    if mode in ("TI", "hybrid"):
        try:
            from ti_fields import compute_group_amplitudes, compute_modulation_depth
            freq1 = ga_result.get("freq_carrier", 1000.0)
            freq2 = freq1 + ga_result.get("delta_freq", 40.0)
            A1, A2 = compute_group_amplitudes(amps, group, basis_matrix, freq1, freq2)
            M = compute_modulation_depth(A1, A2)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(M, "b-", linewidth=0.5, alpha=0.7)
            ax.axvline(target_idx, color="red", linestyle="--", label=f"Target (idx={target_idx})")
            if len(surface_indices) > 0:
                ax.axhline(np.max(M[surface_indices]), color="orange", linestyle=":", alpha=0.5,
                           label=f"Surface max = {np.max(M[surface_indices]):.4f}")
            ax.set_xlabel("Element index")
            ax.set_ylabel("Modulation depth M(r)")
            ax.set_title(f"TI Modulation Depth — Target M = {M[target_idx]:.4f}")
            ax.legend()
            ax.grid(True, alpha=0.2)
            p = out_dir / "ti_modulation_depth.png"
            fig.tight_layout()
            fig.savefig(p, dpi=200)
            plt.close(fig)
            plot_paths.append(str(p))
        except Exception:
            pass

    # Plot NTS V_peak
    if mode in ("NTS", "hybrid"):
        try:
            from nts_timing import compute_v_peak
            from config import paper_baseline as _pb
            _nts = _pb().nts
            fire_times = np.array(ga_result.get("fire_times", np.zeros(len(amps))))
            V_peak = compute_v_peak(amps, fire_times, basis_matrix,
                                    tau_m=_nts.tau_m_s, q_pulse=_nts.q_pulse)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(V_peak, "g-", linewidth=0.5, alpha=0.7)
            ax.axvline(target_idx, color="red", linestyle="--", label=f"Target (idx={target_idx})")
            if len(surface_indices) > 0:
                ax.axhline(np.max(V_peak[surface_indices]), color="orange", linestyle=":", alpha=0.5,
                           label=f"Surface max = {np.max(V_peak[surface_indices]):.4f}")
            ax.set_xlabel("Element index")
            ax.set_ylabel("V_peak (a.u.)")
            ax.set_title(f"NTS V_peak — Target V = {V_peak[target_idx]:.4f}")
            ax.legend()
            ax.grid(True, alpha=0.2)
            p = out_dir / "nts_v_peak.png"
            fig.tight_layout()
            fig.savefig(p, dpi=200)
            plt.close(fig)
            plot_paths.append(str(p))
        except Exception:
            pass

    # Plot NTS firing schedule
    if mode in ("NTS", "hybrid") and "fire_times" in ga_result:
        try:
            fire_times = np.array(ga_result["fire_times"])
            n = len(fire_times)
            fig, ax = plt.subplots(figsize=(12, max(4, n * 0.3)))
            order = np.argsort(fire_times)
            for rank, coil_idx in enumerate(order):
                t = fire_times[coil_idx] * 1e3
                a = amps[coil_idx]
                ax.barh(rank, width=0.1, left=t, height=0.6, color="steelblue", alpha=0.8)
                ax.text(t + 0.12, rank, f"C{coil_idx} (α={a:.2f})", va="center", fontsize=7)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Firing order")
            ax.set_title("NTS Firing Schedule")
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"#{i + 1}" for i in range(n)])
            p = out_dir / "nts_firing_schedule.png"
            fig.tight_layout()
            fig.savefig(p, dpi=200)
            plt.close(fig)
            plot_paths.append(str(p))
        except Exception:
            pass

    # Plot amplitude distribution
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1, ax2 = axes
        ax1.bar(range(len(amps)), amps, color="steelblue")
        ax1.set_xlabel("Coil index")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Optimized Amplitudes")
        ax1.grid(True, axis="y", alpha=0.2)

        colors = ["#2196F3" if g < 0.5 else "#FF5722" for g in group]
        ax2.bar(range(len(group)), group, color=colors)
        ax2.set_xlabel("Coil index")
        ax2.set_ylabel("Group")
        ax2.set_title("TI Group Assignment (0=blue, 1=red)")
        ax2.set_yticks([0, 1])
        ax2.grid(True, axis="y", alpha=0.2)

        p = out_dir / "amplitude_and_groups.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))
    except Exception:
        pass

    return plot_paths


def build_sensitivity_plots(out_dir: Path, pipeline_dir: Path) -> list[str]:
    """Generate sensitivity analysis plots from pipeline output."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_paths: list[str] = []
    sens_path = pipeline_dir / "sensitivity_analysis.npz"
    if not sens_path.exists():
        return plot_paths

    try:
        data = np.load(str(sens_path), allow_pickle=True)
    except Exception:
        return plot_paths

    # --- Plot 1: Jacobian heatmap (TI) ---
    key_ti = "jacobian_ti__J_full"
    if key_ti in data:
        J = data[key_ti]
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(np.abs(J), aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Coil index")
        ax.set_ylabel("Output metric")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["M_target", "M_surface_max", "SAR_max"])
        ax.set_title("TI Jacobian |∂y/∂α| — which coils matter most")
        fig.colorbar(im, ax=ax, label="|∂y/∂αᵢ|")
        p = out_dir / "jacobian_heatmap_ti.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 2: SVD singular values (TI) ---
    svd_key = "svd_ti__S"
    if svd_key in data:
        S = data[svd_key]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(S)), S, color="steelblue")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Singular value")
        ax.set_title("TI Jacobian Singular Values")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        p = out_dir / "svd_singular_values_ti.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 3: Reachable set scatter + hull (TI) ---
    rt_t = "reachable_ti__M_target"
    rt_s = "reachable_ti__M_surface_max"
    rt_h = "reachable_ti__hull_vertices"
    if rt_t in data and rt_s in data:
        Mt = data[rt_t]
        Ms = data[rt_s]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Mt, Ms, s=5, alpha=0.3, color="steelblue", label="Samples")
        if rt_h in data:
            hull = data[rt_h]
            # Close the hull polygon
            hull_closed = np.vstack([hull, hull[0:1]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], "r-", linewidth=1.5, label="Convex hull")
        ax.set_xlabel("M_target (V/m)")
        ax.set_ylabel("M_surface_max (V/m)")
        ax.set_title("TI Reachable Set — Depth vs Selectivity")
        ax.legend()
        ax.grid(True, alpha=0.2)
        p = out_dir / "reachable_set_ti.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 4: Pareto front (TI) ---
    pt_t = "pareto_ti__M_target"
    pt_s = "pareto_ti__M_surface_max"
    pt_d = "pareto_ti__is_dominated"
    if pt_t in data and pt_s in data:
        Mt = data[pt_t]
        Ms = data[pt_s]
        dom = data[pt_d] if pt_d in data else np.zeros(len(Mt), dtype=bool)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Mt[dom], Ms[dom], s=20, alpha=0.3, color="gray", label="Dominated")
        ax.scatter(Mt[~dom], Ms[~dom], s=40, alpha=0.8, color="red", label="Pareto front")
        # Sort Pareto points for line
        pareto_pts = np.column_stack([Mt[~dom], Ms[~dom]])
        if len(pareto_pts) > 1:
            order = np.argsort(pareto_pts[:, 0])
            ax.plot(pareto_pts[order, 0], pareto_pts[order, 1], "r-", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("M_target (V/m)")
        ax.set_ylabel("M_surface_max (V/m)")
        ax.set_title("TI Pareto Front — Depth-Selectivity Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.2)
        p = out_dir / "pareto_front_ti.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 5: Reachable set (NTS) ---
    rn_t = "reachable_nts__V_target"
    rn_s = "reachable_nts__V_surface_max"
    rn_h = "reachable_nts__hull_vertices"
    if rn_t in data and rn_s in data:
        Vt = data[rn_t]
        Vs = data[rn_s]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Vt, Vs, s=5, alpha=0.3, color="forestgreen", label="Samples")
        if rn_h in data:
            hull = data[rn_h]
            hull_closed = np.vstack([hull, hull[0:1]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], "r-", linewidth=1.5, label="Convex hull")
        ax.set_xlabel("V_target (a.u.)")
        ax.set_ylabel("V_surface_max (a.u.)")
        ax.set_title("NTS Reachable Set — Depth vs Selectivity")
        ax.legend()
        ax.grid(True, alpha=0.2)
        p = out_dir / "reachable_set_nts.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 6: Pareto front (NTS) ---
    pn_t = "pareto_nts__V_target"
    pn_s = "pareto_nts__V_surface_max"
    pn_d = "pareto_nts__is_dominated"
    if pn_t in data and pn_s in data:
        Vt = data[pn_t]
        Vs = data[pn_s]
        dom = data[pn_d] if pn_d in data else np.zeros(len(Vt), dtype=bool)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Vt[dom], Vs[dom], s=20, alpha=0.3, color="gray", label="Dominated")
        ax.scatter(Vt[~dom], Vs[~dom], s=40, alpha=0.8, color="darkgreen", label="Pareto front")
        pareto_pts = np.column_stack([Vt[~dom], Vs[~dom]])
        if len(pareto_pts) > 1:
            order = np.argsort(pareto_pts[:, 0])
            ax.plot(pareto_pts[order, 0], pareto_pts[order, 1], "g-", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("V_target (a.u.)")
        ax.set_ylabel("V_surface_max (a.u.)")
        ax.set_title("NTS Pareto Front — Depth-Selectivity Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.2)
        p = out_dir / "pareto_front_nts.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    return plot_paths


def build_cp_bridge_plots(out_dir: Path, pipeline_dir: Path) -> list[str]:
    """Generate CP bridge analysis plots from pipeline output."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_paths: list[str] = []
    cp_path = pipeline_dir / "cp_bridge_analysis.npz"
    if not cp_path.exists():
        return plot_paths

    try:
        data = np.load(str(cp_path), allow_pickle=True)
    except Exception:
        return plot_paths

    # --- Plot 1: Transfer entropy heatmap ---
    if "te_matrix" in data:
        te = data["te_matrix"]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(te, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Coil j (target)")
        ax.set_ylabel("Coil i (source)")
        ax.set_title("Transfer Entropy T_{i→j} (Directed Information Flow)")
        fig.colorbar(im, ax=ax, label="T_{i→j}")
        p = out_dir / "cp_transfer_entropy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 2: Energy breakdown bar chart ---
    energy_keys = ["E_nca", "E_mf", "E_arch", "E_phi", "E_couple", "E_morl"]
    energy_vals = []
    energy_labels = []
    for k in energy_keys:
        ekey = f"energy__{k}"
        if ekey in data:
            energy_vals.append(float(data[ekey]))
            energy_labels.append(k.replace("E_", ""))

    if energy_vals:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336", "#607D8B"]
        bars = ax.bar(energy_labels, energy_vals, color=colors_bar[:len(energy_labels)])
        ax.set_ylabel("Energy")
        ax.set_title("4D NCA Energy Functional Breakdown")
        ax.grid(True, axis="y", alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, energy_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        p = out_dir / "cp_energy_breakdown.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 3: Agent free energy distribution ---
    if "agent_free_energies" in data:
        fe = data["agent_free_energies"]
        fig, ax = plt.subplots(figsize=(10, 4))
        N = len(fe)
        # Color by M/F group if available
        if "group" in data:
            grp = data["group"]
            colors_agent = ["#2196F3" if g < 0.5 else "#FF5722" for g in grp]
        else:
            colors_agent = ["steelblue"] * N
        ax.bar(range(N), fe, color=colors_agent)
        ax.set_xlabel("Coil index")
        ax.set_ylabel("Free energy F^i")
        ax.set_title("Agent Free Energies (blue=M, red=F)")
        ax.grid(True, axis="y", alpha=0.2)
        p = out_dir / "cp_agent_free_energies.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 4: World map (Pareto + Φ + probability) ---
    if "world_energies" in data and "world_coherences" in data and "world_probabilities" in data:
        energies = data["world_energies"]
        coherences = data["world_coherences"]
        probs = data["world_probabilities"]

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(energies, coherences,
                             s=probs * 500 + 10,
                             c=probs, cmap="plasma",
                             alpha=0.7, edgecolors="black", linewidth=0.5)
        fig.colorbar(scatter, ax=ax, label="P(world)")
        ax.set_xlabel("E_total (total energy)")
        ax.set_ylabel("Φ^{collective} (coherence)")
        ax.set_title("Many Worlds — Energy vs Coherence (size ∝ probability)")
        ax.grid(True, alpha=0.2)
        p = out_dir / "cp_world_map.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # --- Plot 5: MORL objectives radar/bar ---
    morl_keys = ["morl__J_phi", "morl__J_arch", "morl__J_sync", "morl__J_task"]
    morl_vals = []
    morl_labels = []
    for k in morl_keys:
        if k in data:
            morl_vals.append(float(data[k]))
            morl_labels.append(k.replace("morl__", ""))

    if morl_vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(morl_labels, morl_vals, color=["#9C27B0", "#4CAF50", "#2196F3", "#FF9800"])
        ax.set_xlabel("Objective value")
        ax.set_title("MORL Objectives")
        ax.grid(True, axis="x", alpha=0.2)
        ax.axvline(0, color="black", linewidth=0.5)
        p = out_dir / "cp_morl_objectives.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    return plot_paths


def build_trajectory_plots(out_dir: Path, pipeline_dir: Path) -> list[str]:
    """Generate trajectory visualisation plots.

    Produces up to 5 plots from trajectory_result.npz:
        1. Amplitude time-series
        2. Output evolution (M_target, M_surface, SAR)
        3. Energy landscape
        4. Phase-space trajectory (Φ vs time)
        5. Cognitive H-theorem trace (H_cog vs time)

    Parameters
    ----------
    out_dir : Path
    pipeline_dir : Path

    Returns
    -------
    plot_paths : list of str
    """
    traj_path = pipeline_dir / "trajectory_result.npz"
    if not traj_path.exists():
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    data = dict(np.load(traj_path, allow_pickle=True))
    plot_paths: list[str] = []

    time_arr = data.get("time", np.array([]))
    amps = data.get("amplitudes", np.array([]))
    outputs = data.get("outputs", np.array([]))
    energy = data.get("energy", np.array([]))
    phi = data.get("phi", np.array([]))
    sync = data.get("sync", np.array([]))
    h_cog = data.get("h_cog", np.array([]))
    wp_idx = data.get("waypoint_indices", np.array([]))

    if len(time_arr) == 0:
        return []

    # Plot 1: Amplitude time-series
    if amps.ndim == 2 and amps.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        N = amps.shape[1]
        for j in range(N):
            ax.plot(time_arr, amps[:, j], alpha=0.6, linewidth=0.8)
        # Mark waypoints
        for wi in wp_idx:
            if wi < len(time_arr):
                ax.axvline(time_arr[wi], color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Coil Amplitudes Along Trajectory ({N} coils)")
        ax.grid(True, alpha=0.2)
        p = out_dir / "traj_amplitudes.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 2: Output evolution
    if outputs.ndim == 2 and outputs.shape[1] >= 4:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channels = [
            (0, "target_metric", "#2196F3"),
            (1, "surface_metric", "#FF9800"),
            (3, "sar_max", "#F44336"),
        ]
        for ax, (idx, label, colour) in zip(axes, channels):
            ax.plot(time_arr, outputs[:, idx], color=colour, linewidth=1.5)
            for wi in wp_idx:
                if wi < len(time_arr):
                    ax.axvline(time_arr[wi], color="red", linestyle="--", alpha=0.3, linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Normalised time")
        axes[0].set_title("Outputs Along Trajectory")
        p = out_dir / "traj_outputs.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 3: Energy landscape
    if len(energy) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_arr, energy, color="#9C27B0", linewidth=1.5)
        for wi in wp_idx:
            if wi < len(time_arr):
                ax.axvline(time_arr[wi], color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("E_total")
        ax.set_title("4D NCA Energy Along Trajectory")
        ax.grid(True, alpha=0.2)
        p = out_dir / "traj_energy.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 4: Φ and R vs time
    if len(phi) > 0 and len(sync) > 0:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(time_arr, phi, color="#4CAF50", linewidth=1.5, label="Φ")
        ax1.set_xlabel("Normalised time")
        ax1.set_ylabel("Φ (collective)", color="#4CAF50")
        ax1.tick_params(axis="y", labelcolor="#4CAF50")

        ax2 = ax1.twinx()
        ax2.plot(time_arr, sync, color="#E91E63", linewidth=1.5, label="R (sync)")
        ax2.set_ylabel("R (sync order)", color="#E91E63")
        ax2.tick_params(axis="y", labelcolor="#E91E63")

        for wi in wp_idx:
            if wi < len(time_arr):
                ax1.axvline(time_arr[wi], color="red", linestyle="--", alpha=0.3, linewidth=0.8)

        ax1.set_title("Collective Φ and Sync R Along Trajectory")
        ax1.grid(True, alpha=0.2)
        p = out_dir / "traj_phi_sync.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    # Plot 5: H_cog trace
    if len(h_cog) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_arr, h_cog, color="#6A1B9A", linewidth=1.5, label="H_cog")
        if len(h_cog) > 1:
            dH = np.diff(h_cog)
            uphill = np.where(dH > 1e-6)[0]
            if len(uphill) > 0:
                ax.scatter(time_arr[uphill + 1], h_cog[uphill + 1],
                           color="#D32F2F", s=15, label="dH/dt > 0")
        for wi in wp_idx:
            if wi < len(time_arr):
                ax.axvline(time_arr[wi], color="red", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Normalised time")
        ax.set_ylabel("H_cog")
        ax.set_title("Cognitive H-Theorem Trace")
        ax.grid(True, alpha=0.2)
        ax.legend()
        p = out_dir / "traj_h_cog.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        plot_paths.append(str(p))

    return plot_paths


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    sim_root = Path(args.sim_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(sim_root)
    runs = collect_runs(summaries, fallback_goal_e=args.goal_e_vpm)
    status = build_status(
        summaries=summaries,
        runs=runs,
        fallback_goal_e=args.goal_e_vpm,
        tol_frac=args.goal_tolerance_frac,
    )

    status_json = out_dir / "project_status.json"
    status_json.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    status_md = out_dir / "project_status.md"
    write_markdown(status_md, status)
    position_status = build_position_status(
        runs=runs, fallback_goal_e=args.goal_e_vpm, tol_frac=args.goal_tolerance_frac
    )
    position_status_json = out_dir / "position_status.json"
    position_status_json.write_text(
        json.dumps(position_status, indent=2, sort_keys=True), encoding="utf-8"
    )
    position_status_md = out_dir / "position_status.md"
    write_position_markdown(position_status_md, position_status)

    plot_paths = build_plots(out_dir, runs, args.goal_e_vpm)
    position_plot_paths = build_position_plots(out_dir, runs, args.goal_e_vpm)

    print(f"Simulation root: {sim_root}")
    print(f"Summaries found: {len(summaries)}")
    print(f"Runs found: {len(runs)}")
    print(f"Wrote status JSON: {status_json}")
    print(f"Wrote status Markdown: {status_md}")
    print(f"Wrote position status JSON: {position_status_json}")
    print(f"Wrote position status Markdown: {position_status_md}")
    if plot_paths:
        print("Wrote plots:")
        for p in plot_paths:
            print(f"  - {p}")
    else:
        print("Plot generation skipped (no valid runs or matplotlib unavailable).")
    if position_plot_paths:
        print("Wrote position plots:")
        for p in position_plot_paths:
            print(f"  - {p}")
    else:
        print("Position plot generation skipped (no valid position-tagged runs).")

    # Pipeline-specific dashboard
    here = Path(__file__).resolve().parent
    pipeline_dir = Path(args.pipeline_dir) if args.pipeline_dir else here / "pipeline_output"
    if pipeline_dir.exists():
        print("\n--- Pipeline Results Dashboard ---")
        metrics = compute_pipeline_metrics(pipeline_dir)
        metrics_json = out_dir / "pipeline_metrics.json"
        metrics_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(f"Wrote pipeline metrics: {metrics_json}")

        pipeline_md = out_dir / "pipeline_dashboard.md"
        write_pipeline_markdown(pipeline_md, metrics)
        print(f"Wrote pipeline dashboard: {pipeline_md}")

        pipeline_plots = build_pipeline_plots(out_dir, pipeline_dir)
        if pipeline_plots:
            print("Wrote pipeline plots:")
            for p in pipeline_plots:
                print(f"  - {p}")
        else:
            print("Pipeline plot generation skipped.")

        # Sensitivity analysis plots
        sens_plots = build_sensitivity_plots(out_dir, pipeline_dir)
        if sens_plots:
            print("Wrote sensitivity plots:")
            for p in sens_plots:
                print(f"  - {p}")

        # CP Bridge plots
        cp_plots = build_cp_bridge_plots(out_dir, pipeline_dir)
        if cp_plots:
            print("Wrote CP bridge plots:")
            for p in cp_plots:
                print(f"  - {p}")

        # Trajectory plots
        traj_plots = build_trajectory_plots(out_dir, pipeline_dir)
        if traj_plots:
            print("Wrote trajectory plots:")
            for p in traj_plots:
                print(f"  - {p}")

        # Print sensitivity summary if available
        sens_summary_path = pipeline_dir / "sensitivity_summary.txt"
        if sens_summary_path.exists():
            print("\n--- Sensitivity Analysis Summary ---")
            print(sens_summary_path.read_text(encoding="utf-8"))

        # Print CP bridge summary if available
        cp_summary_path = pipeline_dir / "cp_bridge_summary.txt"
        if cp_summary_path.exists():
            print("\n--- CP Bridge Analysis Summary ---")
            print(cp_summary_path.read_text(encoding="utf-8"))
    else:
        print("\nNo pipeline_output/ found — run run_pipeline.py first for TI/NTS metrics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
