"""Basis field computation and caching for the Omnidream TMS array.

Precomputes per-coil E-fields via SimNIBS and stores them as a reusable
basis matrix.  During optimisation (GA / SAC), all field evaluations reduce
to fast matrix–vector products against this cached matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import OmnidreamConfig, SimConfig


# ---------------------------------------------------------------------------
# Full E-field vector extraction
# ---------------------------------------------------------------------------

def extract_full_efield(
    result_mesh_path: str | Path,
    gm_tag: int = 2,
) -> np.ndarray:
    """Extract the full E-field magnitude vector on grey-matter elements.

    Unlike ``field_calculator.extract_field_metrics`` which returns summary
    statistics, this returns the raw per-element |E| array.

    Parameters
    ----------
    result_mesh_path : path
        Path to a SimNIBS result ``.msh`` file.
    gm_tag : int
        Tissue tag for grey matter.

    Returns
    -------
    e_gm : ndarray of shape (num_gm_elements,)
        |E| in V/m for each grey-matter element.
    """
    from simnibs.mesh_tools import mesh_io

    mesh = mesh_io.read_msh(str(result_mesh_path))

    if "magnE" in mesh.field:
        e_mag = np.asarray(mesh.field["magnE"].value, dtype=float)
    elif "E" in mesh.field:
        E_vec = np.asarray(mesh.field["E"].value, dtype=float)
        e_mag = np.linalg.norm(E_vec, axis=1)
    else:
        raise KeyError(f"No E-field found in {result_mesh_path}")

    tags = np.asarray(mesh.elm.tag2, dtype=int)
    gm_mask = tags == gm_tag
    if not np.any(gm_mask):
        # Fall back to full mesh if GM tag not found
        gm_mask = np.ones_like(tags, dtype=bool)

    return e_mag[gm_mask]


# ---------------------------------------------------------------------------
# Sample-point index helpers
# ---------------------------------------------------------------------------

def get_sample_point_indices(
    result_mesh_path: str | Path,
    target_coords_mm: np.ndarray,
    gm_tag: int = 2,
    surface_depth_mm: float = 3.0,
) -> tuple[int, np.ndarray]:
    """Find target and surface element indices in the grey-matter mesh.

    Parameters
    ----------
    result_mesh_path : path
        Path to a SimNIBS result ``.msh`` file.
    target_coords_mm : ndarray (3,)
        Target location in mm.
    gm_tag : int
        Grey-matter tissue tag.
    surface_depth_mm : float
        Elements within this distance of the outermost GM surface are
        considered "surface" elements.

    Returns
    -------
    target_idx : int
        Index of the GM element closest to the target.
    surface_indices : ndarray of int
        Indices of GM elements near the surface.
    """
    from simnibs.mesh_tools import mesh_io

    mesh = mesh_io.read_msh(str(result_mesh_path))
    tags = np.asarray(mesh.elm.tag2, dtype=int)
    gm_mask = tags == gm_tag
    if not np.any(gm_mask):
        gm_mask = np.ones_like(tags, dtype=bool)

    # Element centroids
    centroids = mesh.elements_baricenters().value
    gm_centroids = centroids[gm_mask]

    # Target: closest element
    target = np.asarray(target_coords_mm, float)
    dists_to_target = np.linalg.norm(gm_centroids - target, axis=1)
    target_idx = int(np.argmin(dists_to_target))

    # Surface: elements closest to the outer boundary
    # Approximate by max-z elements (near scalp) or outermost radial distance
    radii = np.linalg.norm(gm_centroids, axis=1)
    max_radius = np.max(radii)
    surface_mask = radii >= (max_radius - surface_depth_mm)
    surface_indices = np.where(surface_mask)[0]

    if len(surface_indices) == 0:
        # Fallback: top 10% by radius
        threshold = np.percentile(radii, 90)
        surface_indices = np.where(radii >= threshold)[0]

    return target_idx, surface_indices


# ---------------------------------------------------------------------------
# Config hashing for cache keys
# ---------------------------------------------------------------------------

def _config_hash(config: OmnidreamConfig, positions: np.ndarray, orientations: np.ndarray) -> str:
    """Deterministic hash of the configuration + geometry for cache keying."""
    h = hashlib.sha256()

    # Include key config fields
    cfg_str = json.dumps({
        "coil_turns": config.coil.total_turns,
        "winding_w": config.coil.winding_width_mm,
        "winding_h": config.coil.winding_height_mm,
        "gap": config.coil.base_gap_mm,
        "didt_ref": config.sim.didt_reference_as,
        "gm_tag": config.sim.gm_tag,
        "head_mesh": config.sim.head_mesh_path,
        "coil_file": config.sim.coil_file_path,
    }, sort_keys=True)
    h.update(cfg_str.encode())

    # Include geometry
    h.update(positions.tobytes())
    h.update(orientations.tobytes())

    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Single-coil basis field computation
# ---------------------------------------------------------------------------

def compute_basis_field(
    head_mesh: str | Path,
    coil_file: str | Path,
    position: np.ndarray,
    orientation: np.ndarray,
    didt_ref: float,
    gm_tag: int = 2,
    output_root: str | Path = "simulations/basis",
) -> np.ndarray:
    """Compute the E-field from a single coil at a reference dI/dt.

    Parameters
    ----------
    head_mesh : path
        Path to SimNIBS head mesh.
    coil_file : path
        Path to .tcd/.ccd coil file.
    position : ndarray (3,)
        Coil centre position (mm or EEG label handled upstream).
    orientation : ndarray (3,)
        Coil y-direction target (mm or EEG label).
    didt_ref : float
        Reference dI/dt in A/s.
    gm_tag : int
        Grey-matter tissue tag.
    output_root : path
        Root directory for simulation output.

    Returns
    -------
    e_field : ndarray (num_gm_elements,)
        |E| on grey-matter elements at the reference dI/dt.
    """
    from field_calculator import run_one

    metrics = run_one(
        head_mesh=Path(head_mesh),
        coil_file=Path(coil_file),
        simulations_root=Path(output_root),
        centre=position.tolist(),
        pos_ydir=(position + orientation).tolist(),
        didt_as=didt_ref,
        gm_tag=gm_tag,
    )

    result_mesh_path = metrics["result_mesh"]
    return extract_full_efield(result_mesh_path, gm_tag)


# ---------------------------------------------------------------------------
# Full basis matrix computation
# ---------------------------------------------------------------------------

def compute_all_basis_fields(
    config: OmnidreamConfig,
    positions: np.ndarray,
    orientations: np.ndarray,
    use_cache: bool = True,
    parallel: bool = False,
) -> np.ndarray:
    """Compute or load the full basis matrix for all coils.

    Parameters
    ----------
    config : OmnidreamConfig
    positions : ndarray (N, 3)
        Coil positions in mm.
    orientations : ndarray (N, 3)
        Coil y-direction vectors.
    use_cache : bool
        If True, attempt to load from / save to the basis cache.
    parallel : bool
        If True, use multiprocessing (not yet implemented — sequential fallback).

    Returns
    -------
    basis_matrix : ndarray (num_gm_elements, N)
        Each column is the |E| field from one coil at reference dI/dt.
    """
    cache_dir = Path(config.sim.basis_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _config_hash(config, positions, orientations)
    cache_path = cache_dir / f"{cache_key}.npz"

    if use_cache and cache_path.exists():
        data = np.load(str(cache_path))
        return data["basis_matrix"]

    N = len(positions)
    columns = []
    for i in range(N):
        print(f"  [basis field {i + 1}/{N}] position={positions[i]}")
        col = compute_basis_field(
            head_mesh=config.sim.head_mesh_path,
            coil_file=config.sim.coil_file_path,
            position=positions[i],
            orientation=orientations[i],
            didt_ref=config.sim.didt_reference_as,
            gm_tag=config.sim.gm_tag,
            output_root=config.sim.output_root,
        )
        columns.append(col)

    basis_matrix = np.column_stack(columns)

    if use_cache:
        np.savez_compressed(str(cache_path), basis_matrix=basis_matrix)
        print(f"  Cached basis matrix → {cache_path}")

    return basis_matrix


# ---------------------------------------------------------------------------
# Synthetic basis fields (for testing without SimNIBS)
# ---------------------------------------------------------------------------

def generate_synthetic_basis(
    n_points: int = 5000,
    n_coils: int = 32,
    target_idx: int = 2500,
    surface_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Generate a synthetic basis matrix for testing without SimNIBS.

    Each column is a Gaussian-like field peaked near a random surface point
    and decaying toward the target.

    Parameters
    ----------
    n_points : int
    n_coils : int
    target_idx : int
    surface_fraction : float
        Fraction of points considered "surface" (highest indices).
    rng : Generator, optional

    Returns
    -------
    basis_matrix : ndarray (n_points, n_coils)
    target_idx : int
    surface_indices : ndarray of int
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Surface indices: last surface_fraction of points
    n_surface = max(1, int(n_points * surface_fraction))
    surface_indices = np.arange(n_points - n_surface, n_points)

    basis_matrix = np.zeros((n_points, n_coils))
    for c in range(n_coils):
        # Peak near a random surface point
        peak_idx = rng.integers(n_points - n_surface, n_points)
        sigma = n_points * 0.15
        indices = np.arange(n_points, dtype=float)
        basis_matrix[:, c] = np.exp(-0.5 * ((indices - peak_idx) / sigma) ** 2)
        # Add some noise
        basis_matrix[:, c] += rng.normal(0, 0.01, n_points)
        basis_matrix[:, c] = np.maximum(basis_matrix[:, c], 0)

    return basis_matrix, target_idx, surface_indices
