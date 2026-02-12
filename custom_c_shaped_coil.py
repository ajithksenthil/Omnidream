"""Generate a miniature C-shaped TMS coil in SimNIBS .tcd format.

Paper baseline (Jiang et al., J Neural Eng 2023):
- 30 turns total
- 7 mm x 4 mm winding footprint at each base
- 5 mm shortest base-to-base gap
- 150 degree base angle

This script builds an approximate geometric coil model with two rectangular
windings carrying opposite current directions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from simnibs.simulation.tms_coil.tms_coil import TmsCoil
    from simnibs.simulation.tms_coil.tms_coil_element import LineSegmentElements
    from simnibs.simulation.tms_coil.tms_stimulator import TmsStimulator
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "SimNIBS Python package not found. Run with SimNIBS env Python, e.g. "
        "/Users/ajithsenthil/Applications/SimNIBS-4.5/simnibs_env/bin/python"
    ) from exc


@dataclass
class CoilSpec:
    # Paper-aligned baseline parameters in millimeters.
    total_turns: int = 30
    turns_per_base: int = 15
    winding_width_mm: float = 7.0
    winding_height_mm: float = 4.0
    base_gap_mm: float = 5.0
    base_axis_angle_deg: float = 75.0  # +/-75 -> 150 deg included angle
    turn_pitch_mm: float = 0.20  # paper FEM used 0.2 mm conductor diameter
    layers_per_base: int = 2
    layer_spacing_mm: float = 0.25
    segments_per_edge: int = 24
    z_offset_mm: float = -2.0
    # Keep above calibrated operating point (~6.87e8 A/s).
    stimulator_max_didt_as: float = 1.0e9
    casing_distance_mm: float = 1.0
    casing_thickness_mm: float = 0.5
    # Field sampling bounds used by SimNIBS when exporting coil fields.
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-25.0, 25.0),
        (-25.0, 25.0),
        (-15.0, 15.0),
    )
    resolution_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _rotate_2d(points_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points_xy @ rot.T


def _edge_linspace(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    # Exclude endpoint to avoid duplicated corner vertices.
    return np.linspace(a, b, n, endpoint=False, dtype=float)


def _rectangle_loop_points(
    center_xy: np.ndarray,
    width_mm: float,
    height_mm: float,
    angle_deg: float,
    segments_per_edge: int,
    clockwise: bool,
    z_mm: float,
) -> np.ndarray:
    half_w = width_mm / 2.0
    half_h = height_mm / 2.0

    corners = np.array(
        [
            [half_w, half_h],
            [half_w, -half_h],
            [-half_w, -half_h],
            [-half_w, half_h],
        ],
        dtype=float,
    )
    if not clockwise:
        corners = corners[::-1]

    corners = _rotate_2d(corners, angle_deg) + center_xy

    segments = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        segments.append(_edge_linspace(a, b, segments_per_edge))

    loop_xy = np.vstack(segments)
    z = np.full((loop_xy.shape[0], 1), z_mm, dtype=float)
    return np.hstack((loop_xy, z))


def _build_base_winding_elements(
    stimulator: TmsStimulator,
    center_xy: np.ndarray,
    angle_deg: float,
    turns: int,
    outer_width_mm: float,
    outer_height_mm: float,
    turn_pitch_mm: float,
    layers_per_base: int,
    layer_spacing_mm: float,
    segments_per_edge: int,
    clockwise: bool,
    z_mm: float,
    base_name: str,
) -> list[LineSegmentElements]:
    elements: list[LineSegmentElements] = []
    if layers_per_base < 1:
        raise ValueError("layers_per_base must be >= 1.")

    for turn_idx in range(turns):
        radial_idx = turn_idx // layers_per_base
        layer_idx = turn_idx % layers_per_base
        width = outer_width_mm - 2.0 * radial_idx * turn_pitch_mm
        height = outer_height_mm - 2.0 * radial_idx * turn_pitch_mm
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Turn geometry collapsed at turn {turn_idx + 1}. "
                "Reduce turns_per_base, increase layers_per_base, or reduce turn_pitch_mm."
            )

        loop_points = _rectangle_loop_points(
            center_xy=center_xy,
            width_mm=width,
            height_mm=height,
            angle_deg=angle_deg,
            segments_per_edge=segments_per_edge,
            clockwise=clockwise,
            z_mm=z_mm + layer_idx * layer_spacing_mm,
        )
        elements.append(
            LineSegmentElements(
                stimulator,
                loop_points,
                name=f"{base_name}_turn_{turn_idx + 1:02d}",
            )
        )
    return elements


def build_c_shaped_coil(spec: CoilSpec) -> tuple[TmsCoil, dict]:
    if spec.turns_per_base * 2 != spec.total_turns:
        raise ValueError("total_turns must equal 2 * turns_per_base for this model.")
    if spec.segments_per_edge < 2:
        raise ValueError("segments_per_edge must be >= 2.")

    stim = TmsStimulator(
        "Miniature C-shaped stimulator",
        "Omnidream",
        max_di_dt=spec.stimulator_max_didt_as,
    )

    left_center = np.array([-spec.base_gap_mm / 2.0, 0.0], dtype=float)
    right_center = np.array([spec.base_gap_mm / 2.0, 0.0], dtype=float)

    # Opposite winding directions emulate opposite currents under equal dI/dt.
    left_elements = _build_base_winding_elements(
        stimulator=stim,
        center_xy=left_center,
        angle_deg=+spec.base_axis_angle_deg,
        turns=spec.turns_per_base,
        outer_width_mm=spec.winding_width_mm,
        outer_height_mm=spec.winding_height_mm,
        turn_pitch_mm=spec.turn_pitch_mm,
        layers_per_base=spec.layers_per_base,
        layer_spacing_mm=spec.layer_spacing_mm,
        segments_per_edge=spec.segments_per_edge,
        clockwise=True,
        z_mm=spec.z_offset_mm,
        base_name="left_base",
    )
    right_elements = _build_base_winding_elements(
        stimulator=stim,
        center_xy=right_center,
        angle_deg=-spec.base_axis_angle_deg,
        turns=spec.turns_per_base,
        outer_width_mm=spec.winding_width_mm,
        outer_height_mm=spec.winding_height_mm,
        turn_pitch_mm=spec.turn_pitch_mm,
        layers_per_base=spec.layers_per_base,
        layer_spacing_mm=spec.layer_spacing_mm,
        segments_per_edge=spec.segments_per_edge,
        clockwise=False,
        z_mm=spec.z_offset_mm,
        base_name="right_base",
    )
    elements = left_elements + right_elements

    coil = TmsCoil(
        elements,
        name="C-shaped miniature TMS coil (paper baseline)",
        brand="Omnidream",
        version="v1.0",
        limits=[list(x) for x in spec.limits],
        resolution=list(spec.resolution_mm),
    )

    # A lightweight casing helps with visualization and collision handling.
    coil.generate_element_casings(
        spec.casing_distance_mm,
        spec.casing_thickness_mm,
        False,
        combined_casing=True,
    )

    all_points = np.vstack([el.get_points(np.eye(4), apply_deformation=False) for el in elements])
    bbox_min = all_points.min(axis=0).tolist()
    bbox_max = all_points.max(axis=0).tolist()
    summary = {
        "spec": asdict(spec),
        "element_count": len(elements),
        "expected_turns": spec.total_turns,
        "bbox_min_mm": bbox_min,
        "bbox_max_mm": bbox_max,
        "left_center_mm": left_center.tolist(),
        "right_center_mm": right_center.tolist(),
    }
    return coil, summary


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parent / "coil_models" / "c_shaped_miniature_v1.tcd"
    )

    p = argparse.ArgumentParser(description="Create C-shaped miniature TMS .tcd coil")
    p.add_argument("--output", default=str(default_output), help="Output .tcd path")
    p.add_argument(
        "--ascii-mode",
        action="store_true",
        help="Write coil arrays in ascii mode (larger files, easier diffing).",
    )
    p.add_argument("--turns-per-base", type=int, default=15)
    p.add_argument("--winding-width-mm", type=float, default=7.0)
    p.add_argument("--winding-height-mm", type=float, default=4.0)
    p.add_argument("--base-gap-mm", type=float, default=5.0)
    p.add_argument("--base-axis-angle-deg", type=float, default=75.0)
    p.add_argument("--turn-pitch-mm", type=float, default=0.20)
    p.add_argument("--layers-per-base", type=int, default=2)
    p.add_argument("--layer-spacing-mm", type=float, default=0.25)
    p.add_argument("--segments-per-edge", type=int, default=24)
    p.add_argument(
        "--stimulator-max-didt-as",
        type=float,
        default=1.0e9,
        help="Metadata max dI/dt (A/s) recorded in the generated coil file.",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    spec = CoilSpec(
        total_turns=args.turns_per_base * 2,
        turns_per_base=args.turns_per_base,
        winding_width_mm=args.winding_width_mm,
        winding_height_mm=args.winding_height_mm,
        base_gap_mm=args.base_gap_mm,
        base_axis_angle_deg=args.base_axis_angle_deg,
        turn_pitch_mm=args.turn_pitch_mm,
        layers_per_base=args.layers_per_base,
        layer_spacing_mm=args.layer_spacing_mm,
        segments_per_edge=args.segments_per_edge,
        stimulator_max_didt_as=args.stimulator_max_didt_as,
    )

    coil, summary = build_c_shaped_coil(spec)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coil.write(str(output_path), ascii_mode=args.ascii_mode)

    summary_path = output_path.with_suffix(".summary.json")
    _write_json(summary_path, summary)

    print(f"Wrote coil: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        f"Elements: {summary['element_count']} (expected turns: {summary['expected_turns']})"
    )
    print(f"BBox mm min={summary['bbox_min_mm']} max={summary['bbox_max_mm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
