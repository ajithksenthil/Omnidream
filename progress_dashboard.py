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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
