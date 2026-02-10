"""Run a position grid search and produce a ranked shortlist.

This script executes many (centre, pos_ydir) combinations at a fixed dI/dt
and ranks them by closeness to a target E-field magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from field_calculator import ensure_path, run_one


def iso_now() -> str:
    return datetime.now().isoformat()


def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_list(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated list.")
    return vals


def slugify(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_").lower()


def append_ndjson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Position grid search for SimNIBS TMS runs")
    p.add_argument("--head-mesh", required=True, help="Path to subject head mesh (.msh)")
    p.add_argument(
        "--coil-file",
        default=str(here / "coil_models" / "c_shaped_miniature_v1.tcd"),
        help="Path to .tcd/.ccd coil file",
    )
    p.add_argument(
        "--output-root",
        default=str(here / "simulations" / "position_grid"),
        help="Root for grid-search sessions",
    )
    p.add_argument(
        "--session-name",
        default="",
        help="Optional fixed session dir name (default: grid_<timestamp>)",
    )
    p.add_argument(
        "--centres",
        default="C3,C4,F3,F4,P3,P4",
        help="Comma-separated EEG labels used as coil centres",
    )
    p.add_argument(
        "--ydirs",
        default="Cz,Fz,Pz",
        help="Comma-separated EEG labels used as pos_ydir targets",
    )
    p.add_argument("--didt", type=float, default=687479035.6513342, help="Absolute dI/dt in A/s")
    p.add_argument("--target-e-vpm", type=float, default=7.2, help="Target E-field (V/m)")
    p.add_argument("--tolerance-frac", type=float, default=0.05, help="Relative error tolerance")
    p.add_argument("--gm-tag", type=int, default=2, help="Gray matter tissue tag")
    p.add_argument("--top-k", type=int, default=10, help="Top-k entries in shortlist")
    p.add_argument("--max-combos", type=int, default=0, help="If >0, cap number of combos")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    head_mesh = ensure_path(args.head_mesh, "Head mesh")
    coil_file = ensure_path(args.coil_file, "Coil file")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    centres = parse_list(args.centres)
    ydirs = parse_list(args.ydirs)

    combos = []
    for c in centres:
        for y in ydirs:
            if c == y:
                continue
            combos.append((c, y))

    if args.max_combos > 0:
        combos = combos[: args.max_combos]

    if not combos:
        raise ValueError("No valid centre/ydir combinations produced.")

    session_name = args.session_name.strip() or f"grid_{ts_now()}"
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    events_path = session_dir / "events.ndjson"
    status_path = session_dir / "status.json"
    raw_path = session_dir / "grid_results.json"
    csv_path = session_dir / "grid_results.csv"
    shortlist_path = session_dir / "shortlist.json"
    shortlist_md = session_dir / "shortlist.md"

    append_ndjson(
        events_path,
        {
            "ts": iso_now(),
            "event": "session_start",
            "session_name": session_name,
            "session_dir": str(session_dir),
            "didt_as": args.didt,
            "target_e_vpm": args.target_e_vpm,
            "combo_count": len(combos),
            "centres": centres,
            "ydirs": ydirs,
        },
    )

    rows: list[dict] = []
    for idx, (centre, ydir) in enumerate(combos, start=1):
        combo_key = f"{centre} -> {ydir}"
        combo_slug = slugify(f"{centre}_{ydir}")
        combo_root = session_dir / f"combo_{combo_slug}"
        combo_root.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(combos)}] {combo_key}")
        append_ndjson(
            events_path,
            {
                "ts": iso_now(),
                "event": "run_start",
                "index": idx,
                "total": len(combos),
                "combo_key": combo_key,
                "centre": centre,
                "pos_ydir": ydir,
            },
        )

        metrics = run_one(
            head_mesh=head_mesh,
            coil_file=coil_file,
            simulations_root=combo_root,
            centre=centre,
            pos_ydir=ydir,
            didt_as=args.didt,
            gm_tag=args.gm_tag,
        )

        abs_err = abs(float(metrics["e_max"]) - args.target_e_vpm)
        rel_err = abs_err / max(abs(args.target_e_vpm), 1e-12)
        hit = rel_err <= args.tolerance_frac
        row = {
            "centre": centre,
            "pos_ydir": ydir,
            "position_key": combo_key,
            "didt_as": args.didt,
            "target_e_vpm": args.target_e_vpm,
            "e_max": float(metrics["e_max"]),
            "e_mean": float(metrics["e_mean"]),
            "e_p99": float(metrics["e_p99"]),
            "e_p999": float(metrics["e_p999"]),
            "abs_error_vpm": abs_err,
            "rel_error_percent": rel_err * 100.0,
            "goal_hit": hit,
            "run_dir": metrics["run_dir"],
            "result_mesh": metrics["result_mesh"],
            "run_error": metrics.get("run_error"),
        }
        rows.append(row)

        best = min(rows, key=lambda r: r["abs_error_vpm"])
        progress_pct = 100.0 * len(rows) / len(combos)
        status = {
            "updated_at": iso_now(),
            "session_name": session_name,
            "session_dir": str(session_dir),
            "didt_as": args.didt,
            "target_e_vpm": args.target_e_vpm,
            "tolerance_fraction": args.tolerance_frac,
            "total_combos": len(combos),
            "completed_combos": len(rows),
            "progress_percent": progress_pct,
            "latest": row,
            "best_so_far": best,
            "goal_hits_so_far": sum(1 for r in rows if r["goal_hit"]),
        }
        write_status(status_path, status)
        append_ndjson(
            events_path,
            {
                "ts": iso_now(),
                "event": "run_end",
                "index": idx,
                "total": len(combos),
                "combo_key": combo_key,
                "e_max": row["e_max"],
                "abs_error_vpm": row["abs_error_vpm"],
                "goal_hit": hit,
                "progress_percent": progress_pct,
            },
        )

    rows.sort(key=lambda r: r["abs_error_vpm"])
    top_k = max(1, min(args.top_k, len(rows)))
    shortlist = rows[:top_k]

    summary = {
        "generated_at": iso_now(),
        "session_name": session_name,
        "session_dir": str(session_dir),
        "head_mesh": str(head_mesh),
        "coil_file": str(coil_file),
        "didt_as": args.didt,
        "target_e_vpm": args.target_e_vpm,
        "tolerance_fraction": args.tolerance_frac,
        "total_combos": len(rows),
        "goal_hits": sum(1 for r in rows if r["goal_hit"]),
        "goal_hit_rate_percent": 100.0 * sum(1 for r in rows if r["goal_hit"]) / len(rows),
        "best": rows[0],
        "shortlist_top_k": top_k,
        "shortlist": shortlist,
        "all_results": rows,
    }

    raw_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_fields = [
        "centre",
        "pos_ydir",
        "position_key",
        "didt_as",
        "target_e_vpm",
        "e_max",
        "e_mean",
        "e_p99",
        "e_p999",
        "abs_error_vpm",
        "rel_error_percent",
        "goal_hit",
        "run_dir",
        "result_mesh",
        "run_error",
    ]
    write_csv(csv_path, rows, csv_fields)

    shortlist_path.write_text(
        json.dumps(
            {
                "generated_at": summary["generated_at"],
                "session_name": session_name,
                "target_e_vpm": args.target_e_vpm,
                "didt_as": args.didt,
                "top_k": top_k,
                "shortlist": shortlist,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Position Grid Shortlist")
    lines.append("")
    lines.append(f"- Session: `{session_name}`")
    lines.append(f"- dI/dt: `{args.didt:.6g}` A/s")
    lines.append(f"- Target E_max: `{args.target_e_vpm}` V/m")
    lines.append(f"- Total combos: `{len(rows)}`")
    lines.append(f"- Goal-hit rate: `{summary['goal_hit_rate_percent']:.2f}%`")
    lines.append("")
    lines.append("## Top Candidates")
    for i, r in enumerate(shortlist, start=1):
        lines.append(
            f"{i}. `{r['position_key']}` | E_max `{r['e_max']:.6g}` | "
            f"error `{r['abs_error_vpm']:.6g}` V/m ({r['rel_error_percent']:.3f}%)"
        )
    lines.append("")
    shortlist_md.write_text("\n".join(lines), encoding="utf-8")

    append_ndjson(
        events_path,
        {
            "ts": iso_now(),
            "event": "session_end",
            "session_name": session_name,
            "raw_path": str(raw_path),
            "shortlist_path": str(shortlist_path),
            "total_combos": len(rows),
        },
    )

    print(f"Session dir: {session_dir}")
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {shortlist_path}")
    print(f"Wrote: {shortlist_md}")
    print(f"Wrote: {status_path}")
    print(f"Wrote: {events_path}")
    print(f"Best: {rows[0]['position_key']} | E_max={rows[0]['e_max']:.6g} | abs_error={rows[0]['abs_error_vpm']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
