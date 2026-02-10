"""Run SimNIBS TMS field simulations and dI/dt calibration sweeps.

This script supports:
1) Running a single simulation with a custom coil file (.tcd/.ccd)
2) Sweeping dI/dt values and extracting field metrics per run

Usage examples:
  python field_calculator.py \
    --head-mesh /path/to/subj.msh \
    --coil-file Omnidream/coil_models/c_shaped_miniature_v1.tcd \
    --sweep-didt "1e6,2e6,5e6,1e7"

  python field_calculator.py \
    --head-mesh /path/to/subj.msh \
    --coil-file Omnidream/coil_models/c_shaped_miniature_v1.tcd \
    --didt 2.0e7
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from simnibs import run_simnibs, sim_struct
    from simnibs.mesh_tools import mesh_io
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "SimNIBS Python package not found. Run with SimNIBS env Python, e.g. "
        "/Users/ajithsenthil/Applications/SimNIBS-4.5/simnibs_env/bin/python"
    ) from exc


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return dt.datetime.now().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_path(path: str | Path, what: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def parse_didt_list(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("No dI/dt values parsed from --sweep-didt.")
    return values


def parse_position(value: str) -> str | list[float]:
    """Accept either EEG labels (e.g. C3) or comma-separated xyz coordinates."""
    if "," not in value:
        return value
    coords = [float(x.strip()) for x in value.split(",")]
    if len(coords) != 3:
        raise ValueError(f"Expected 3 coordinates, got {len(coords)} in '{value}'")
    return coords


def append_ndjson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_session(
    head_mesh: Path,
    coil_file: Path,
    output_dir: Path,
    centre: str | list[float],
    pos_ydir: str | list[float],
    didt_as: float,
) -> sim_struct.SESSION:
    s = sim_struct.SESSION()
    s.fnamehead = str(head_mesh)
    s.pathfem = str(output_dir)

    tms = s.add_tmslist()
    tms.fnamecoil = str(coil_file)

    pos = tms.add_position()
    pos.centre = centre
    pos.pos_ydir = pos_ydir
    pos.didt = didt_as
    return s


def find_result_mesh(output_dir: Path) -> Path:
    scalar = sorted(output_dir.glob("*_scalar.msh"))
    if scalar:
        return scalar[0]
    msh_files = sorted(output_dir.glob("*.msh"))
    if msh_files:
        return msh_files[0]
    raise FileNotFoundError(f"No .msh result file found in {output_dir}")


def _vector_magnitude(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def extract_field_metrics(result_mesh: Path, gm_tag: int) -> dict:
    mesh = mesh_io.read_msh(str(result_mesh))
    field_keys = sorted(mesh.field.keys())

    if "magnE" in mesh.field:
        e_mag = np.asarray(mesh.field["magnE"].value, dtype=float)
    elif "E" in mesh.field:
        e_mag = _vector_magnitude(np.asarray(mesh.field["E"].value, dtype=float))
    else:
        raise KeyError(f"Result missing E-field. Available fields: {field_keys}")

    b_mag = None
    if "magnB" in mesh.field:
        b_mag = np.asarray(mesh.field["magnB"].value, dtype=float)
    elif "B" in mesh.field:
        b_mag = _vector_magnitude(np.asarray(mesh.field["B"].value, dtype=float))

    tags = np.asarray(mesh.elm.tag2, dtype=int)
    gm_mask = tags == gm_tag
    if not np.any(gm_mask):
        gm_mask = np.ones_like(tags, dtype=bool)
        gm_note = f"GM tag {gm_tag} not found. Metrics computed on all elements."
    else:
        gm_note = None

    e_roi = e_mag[gm_mask]
    e_stats = {
        "e_max": float(np.max(e_roi)),
        "e_mean": float(np.mean(e_roi)),
        "e_p99": float(np.percentile(e_roi, 99.0)),
        "e_p999": float(np.percentile(e_roi, 99.9)),
    }

    b_stats = {}
    if b_mag is not None:
        b_roi = b_mag[gm_mask]
        b_stats = {
            "b_max": float(np.max(b_roi)),
            "b_mean": float(np.mean(b_roi)),
            "b_p99": float(np.percentile(b_roi, 99.0)),
        }

    volumes = mesh.elements_volumes_and_areas()
    vol_array = np.asarray(getattr(volumes, "value", volumes), dtype=float)
    v_roi = vol_array[gm_mask]
    e_max = e_stats["e_max"]
    stim_vols = {}
    for frac in (0.25, 0.5, 0.75):
        stim_vols[f"vol_above_{int(frac * 100)}pct_emax"] = float(np.sum(v_roi[e_roi >= frac * e_max]))

    out = {
        "field_keys": field_keys,
        "gm_tag": gm_tag,
        "gm_note": gm_note,
        **e_stats,
        **b_stats,
        **stim_vols,
        "element_count": int(mesh.elm.nr),
        "node_count": int(mesh.nodes.nr),
    }
    return out


def run_one(
    head_mesh: Path,
    coil_file: Path,
    simulations_root: Path,
    centre: str | list[float],
    pos_ydir: str | list[float],
    didt_as: float,
    gm_tag: int,
) -> dict:
    run_dir = simulations_root / f"didt_{didt_as:.6g}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    session = create_session(head_mesh, coil_file, run_dir, centre, pos_ydir, didt_as)

    run_error = None
    try:
        run_simnibs(session)
    except BaseException as exc:  # pragma: no cover - runtime-dependent SimNIBS behavior
        run_error = str(exc)

    try:
        result_mesh = find_result_mesh(run_dir)
    except Exception:
        if run_error is not None:
            raise RuntimeError(
                "SimNIBS failed and no result mesh was produced. "
                f"run_dir={run_dir}"
            ) from None
        raise

    metrics = extract_field_metrics(result_mesh, gm_tag=gm_tag)
    metrics.update(
        {
            "didt_as": didt_as,
            "run_dir": str(run_dir),
            "result_mesh": str(result_mesh),
            "run_error": run_error,
        }
    )

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def fit_linear_calibration(rows: list[dict], x_key: str, y_key: str) -> dict | None:
    xs = np.array([r[x_key] for r in rows], dtype=float)
    ys = np.array([r[y_key] for r in rows], dtype=float)
    if len(xs) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_coil = here / "coil_models" / "c_shaped_miniature_v1.tcd"
    default_runs = here / "simulations"
    parser = argparse.ArgumentParser(description="SimNIBS field simulation + dI/dt calibration sweeps")
    parser.add_argument("--head-mesh", required=True, help="Path to SimNIBS head mesh (.msh)")
    parser.add_argument("--coil-file", default=str(default_coil), help="Path to .tcd/.ccd coil model")
    parser.add_argument("--output-root", default=str(default_runs), help="Simulation output root")
    parser.add_argument(
        "--centre",
        default="C3",
        help='Coil center: EEG label (e.g. "C3") or "x,y,z" in mm',
    )
    parser.add_argument(
        "--pos-ydir",
        default="Cz",
        help='Coil y-direction target: EEG label or "x,y,z" in mm',
    )
    parser.add_argument("--didt", type=float, default=None, help="Single absolute dI/dt value in A/s")
    parser.add_argument(
        "--sweep-didt",
        default="",
        help='Comma-separated absolute dI/dt values in A/s, e.g. "1e6,2e6,5e6"',
    )
    parser.add_argument("--target-e-vpm", type=float, default=7.2, help="Target E-field (V/m) for suggestion")
    parser.add_argument("--target-b-t", type=float, default=0.460, help="Target B-field (Tesla) for suggestion")
    parser.add_argument("--gm-tag", type=int, default=2, help="Tissue tag for gray matter")
    parser.add_argument(
        "--session-name",
        default="",
        help="Optional fixed session directory name under output-root (default: calibration_<timestamp>)",
    )
    parser.add_argument(
        "--event-log",
        default="",
        help="Optional path for progress event log (.ndjson). Default: <batch_dir>/events.ndjson",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Optional path for status JSON. Default: <batch_dir>/status.json",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=0,
        help="Optional expected total runs for progress percentage (default: current run count).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    head_mesh = ensure_path(args.head_mesh, "Head mesh")
    coil_file = ensure_path(args.coil_file, "Coil file")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    centre = parse_position(args.centre)
    pos_ydir = parse_position(args.pos_ydir)

    if not args.sweep_didt and args.didt is None:
        # A practical default sweep for miniature-coil calibration.
        didt_values = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7]
    elif args.sweep_didt:
        didt_values = parse_didt_list(args.sweep_didt)
    else:
        didt_values = [args.didt]

    batch_name = args.session_name.strip() or f"calibration_{timestamp()}"
    batch_dir = output_root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = (
        Path(args.event_log).expanduser().resolve()
        if args.event_log.strip()
        else batch_dir / "events.ndjson"
    )
    status_file_path = (
        Path(args.status_file).expanduser().resolve()
        if args.status_file.strip()
        else batch_dir / "status.json"
    )

    rows = []
    expected_total = args.expected_runs if args.expected_runs > 0 else len(didt_values)
    session_started_at = iso_now()
    append_ndjson(
        event_log_path,
        {
            "ts": iso_now(),
            "event": "session_start",
            "batch_dir": str(batch_dir),
            "batch_name": batch_name,
            "expected_total_runs": expected_total,
            "didt_values_as": didt_values,
        },
    )

    for idx, didt_as in enumerate(didt_values, start=1):
        print(f"[run {idx}/{len(didt_values)}] dI/dt={didt_as:.6g} A/s")
        append_ndjson(
            event_log_path,
            {
                "ts": iso_now(),
                "event": "run_start",
                "batch_dir": str(batch_dir),
                "run_index": idx,
                "run_total": len(didt_values),
                "didt_as": didt_as,
            },
        )
        row = run_one(
            head_mesh=head_mesh,
            coil_file=coil_file,
            simulations_root=batch_dir,
            centre=centre,
            pos_ydir=pos_ydir,
            didt_as=didt_as,
            gm_tag=args.gm_tag,
        )
        rows.append(row)
        print(f"      maxE={row['e_max']:.6g} V/m")
        if "b_max" in row:
            print(f"      maxB={row['b_max']:.6g} T")

        target_e = args.target_e_vpm
        best_row = min(rows, key=lambda r: abs(r["e_max"] - target_e))
        completed = len(rows)
        progress_pct = min(100.0, 100.0 * completed / max(expected_total, 1))
        status_payload = {
            "updated_at": iso_now(),
            "session_started_at": session_started_at,
            "session_name": batch_name,
            "batch_dir": str(batch_dir),
            "target_e_vpm": target_e,
            "target_b_t": args.target_b_t,
            "expected_total_runs": expected_total,
            "completed_runs": completed,
            "progress_percent": progress_pct,
            "latest_run": row,
            "best_run_so_far": {
                "didt_as": best_row["didt_as"],
                "e_max": best_row["e_max"],
                "abs_error_to_target_e": abs(best_row["e_max"] - target_e),
                "run_dir": best_row["run_dir"],
            },
        }
        write_status(status_file_path, status_payload)
        append_ndjson(
            event_log_path,
            {
                "ts": iso_now(),
                "event": "run_end",
                "batch_dir": str(batch_dir),
                "run_index": idx,
                "run_total": len(didt_values),
                "didt_as": didt_as,
                "e_max": row["e_max"],
                "run_error": row.get("run_error"),
                "progress_percent": progress_pct,
            },
        )

    csv_cols = [
        "didt_as",
        "e_max",
        "e_mean",
        "e_p99",
        "e_p999",
        "b_max",
        "b_mean",
        "b_p99",
        "vol_above_25pct_emax",
        "vol_above_50pct_emax",
        "vol_above_75pct_emax",
        "result_mesh",
        "run_dir",
        "gm_note",
    ]
    write_csv(batch_dir / "sweep_metrics.csv", rows, csv_cols)

    summary = {
        "created_at": dt.datetime.now().isoformat(),
        "session_name": batch_name,
        "head_mesh": str(head_mesh),
        "head_mesh_sha256": sha256_file(head_mesh),
        "coil_file": str(coil_file),
        "coil_file_sha256": sha256_file(coil_file),
        "centre": centre,
        "pos_ydir": pos_ydir,
        "event_log_path": str(event_log_path),
        "status_file_path": str(status_file_path),
        "didt_values_as": didt_values,
        "targets": {
            "target_e_vpm": args.target_e_vpm,
            "target_b_t": args.target_b_t,
        },
        "rows": rows,
    }

    fit_e = fit_linear_calibration(rows, "didt_as", "e_max")
    if fit_e is not None:
        summary["fit_e_linear"] = fit_e
        slope = fit_e["slope"]
        intercept = fit_e["intercept"]
        if abs(slope) > 1e-12:
            didt_for_e = (args.target_e_vpm - intercept) / slope
            summary["suggested_didt_for_target_e_as"] = float(didt_for_e)

    if all("b_max" in r for r in rows):
        fit_b = fit_linear_calibration(rows, "didt_as", "b_max")
        if fit_b is not None:
            summary["fit_b_linear"] = fit_b
            slope = fit_b["slope"]
            intercept = fit_b["intercept"]
            if abs(slope) > 1e-12:
                didt_for_b = (args.target_b_t - intercept) / slope
                summary["suggested_didt_for_target_b_as"] = float(didt_for_b)

    with (batch_dir / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    append_ndjson(
        event_log_path,
        {
            "ts": iso_now(),
            "event": "session_end",
            "batch_dir": str(batch_dir),
            "summary_path": str(batch_dir / "calibration_summary.json"),
            "completed_runs": len(rows),
        },
    )

    print(f"\nSaved sweep CSV: {batch_dir / 'sweep_metrics.csv'}")
    print(f"Saved summary: {batch_dir / 'calibration_summary.json'}")
    print(f"Saved status: {status_file_path}")
    print(f"Saved event log: {event_log_path}")
    if "suggested_didt_for_target_e_as" in summary:
        print(
            "Suggested dI/dt for target E "
            f"({args.target_e_vpm} V/m): {summary['suggested_didt_for_target_e_as']:.6g} A/s"
        )
    if "suggested_didt_for_target_b_as" in summary:
        print(
            "Suggested dI/dt for target B "
            f"({args.target_b_t} T): {summary['suggested_didt_for_target_b_as']:.6g} A/s"
        )
    return 0


if __name__ == "__main__":
    # SimNIBS may leave non-daemon threads/process resources alive after runs.
    # Force a clean process termination after writing all artifacts.
    exit_code = main()
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
