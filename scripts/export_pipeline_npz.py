#!/usr/bin/env python3
"""Export Omnidream NPZ artifacts into JSON files for frontend loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

NPZ_ARTIFACTS = (
    "basis_data.npz",
    "sensitivity_analysis.npz",
    "cp_bridge_analysis.npz",
    "trajectory_result.npz",
)


def to_jsonable(value: Any) -> Any:
    """Convert numpy types recursively into JSON-safe builtins."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def export_npz(npz_path: Path, out_path: Path, *, pretty: bool) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: to_jsonable(data[key]) for key in data.files}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2 if pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pipeline_output/*.npz into frontend-friendly JSON files.",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("pipeline_output"),
        help="Directory containing the NPZ artifacts (default: pipeline_output).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for JSON artifacts (default: <pipeline-dir>/web).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON with indentation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    out_dir = (args.out_dir or (pipeline_dir / "web")).resolve()

    if not pipeline_dir.exists():
        raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")

    manifest: dict[str, dict[str, Any]] = {}
    for npz_name in NPZ_ARTIFACTS:
        npz_path = pipeline_dir / npz_name
        if not npz_path.exists():
            print(f"Skipping missing artifact: {npz_path}")
            continue
        json_name = npz_name.replace(".npz", ".json")
        out_path = out_dir / json_name
        payload = export_npz(npz_path, out_path, pretty=args.pretty)
        manifest[npz_name] = {
            "output": str(out_path),
            "keys": sorted(payload.keys()),
        }
        print(f"Exported {npz_path} -> {out_path}")

    manifest_path = out_dir / "pipeline_npz_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
