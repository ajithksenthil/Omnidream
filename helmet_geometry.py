"""Helmet-surface coil placement for the Omnidream concave TMS array.

Provides functions for sampling coil positions on a spherical-cap helmet,
computing inward-pointing orientations, enforcing minimum spacing, and
projecting helmet positions onto an actual scalp surface mesh.

All coordinates are in **millimetres**.
"""

from __future__ import annotations

import numpy as np

from config import HelmetConfig


# ---------------------------------------------------------------------------
# Uniform random sampling on a spherical cap
# ---------------------------------------------------------------------------

def sample_helmet_positions(
    n: int,
    r_inner: float = 90.0,
    theta_max_deg: float = 120.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample *n* positions uniformly on a spherical cap.

    Parameters
    ----------
    n : int
        Number of coil positions.
    r_inner : float
        Helmet inner radius in mm.
    theta_max_deg : float
        Maximum polar angle from the vertex (Cz), in degrees.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    positions : ndarray of shape (n, 3)
        Cartesian coordinates of sampled points on the helmet surface.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_max = np.radians(theta_max_deg)
    cos_theta_max = np.cos(theta_max)

    # Uniform sampling on a spherical cap:
    # cos(θ) is uniform in [cos(θ_max), 1]
    cos_theta = rng.uniform(cos_theta_max, 1.0, size=n)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = r_inner * np.sin(theta) * np.cos(phi)
    y = r_inner * np.sin(theta) * np.sin(phi)
    z = r_inner * np.cos(theta)

    return np.column_stack([x, y, z])


def sample_helmet_positions_from_config(
    cfg: HelmetConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convenience wrapper that reads parameters from a HelmetConfig."""
    return sample_helmet_positions(
        n=cfg.n_coils,
        r_inner=cfg.r_inner_mm,
        theta_max_deg=cfg.theta_max_deg,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Orientations (inward normals)
# ---------------------------------------------------------------------------

def compute_helmet_normals(positions: np.ndarray) -> np.ndarray:
    """Compute inward-pointing unit normal vectors for helmet positions.

    For a sphere centred at the origin, the inward normal at point **p** is
    simply ``-p / |p|``.

    Parameters
    ----------
    positions : ndarray of shape (N, 3)

    Returns
    -------
    normals : ndarray of shape (N, 3)
    """
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return -positions / norms


# ---------------------------------------------------------------------------
# Minimum distance enforcement on curved surface
# ---------------------------------------------------------------------------

def enforce_min_distance_on_sphere(
    positions: np.ndarray,
    min_dist: float,
    r_inner: float,
    theta_max_deg: float = 120.0,
    max_attempts: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Adjust positions so that no two coils are closer than *min_dist* mm.

    Uses Euclidean distance in 3-D (valid when coil footprint is small
    relative to the helmet radius).

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
        Initial positions (modified in place).
    min_dist : float
        Minimum inter-coil distance in mm.
    r_inner : float
        Helmet radius in mm (for re-sampling violating coils).
    theta_max_deg : float
        Angular coverage of the helmet.
    max_attempts : int
        Give up after this many re-sampling iterations.
    rng : numpy Generator, optional

    Returns
    -------
    positions : ndarray of shape (N, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(positions)
    if n < 2:
        return positions

    for _ in range(max_attempts):
        dists = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)
        min_val = np.min(dists)
        if min_val >= min_dist:
            break
        # Replace the second coil of the closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        replacement = sample_helmet_positions(1, r_inner, theta_max_deg, rng)
        positions[j] = replacement[0]

    return positions


# ---------------------------------------------------------------------------
# Scalp-surface projection
# ---------------------------------------------------------------------------

def load_scalp_surface_from_mesh(head_mesh_path: str, scalp_tag: int = 1005) -> np.ndarray:
    """Extract scalp surface node coordinates from a SimNIBS head mesh.

    Parameters
    ----------
    head_mesh_path : str
        Path to a SimNIBS ``.msh`` file.
    scalp_tag : int
        Element tag identifying the scalp surface (default 1005 in SimNIBS 4).

    Returns
    -------
    scalp_points : ndarray of shape (M, 3)
    """
    try:
        from simnibs.mesh_tools import mesh_io
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SimNIBS is required for scalp mesh extraction."
        ) from exc

    mesh = mesh_io.read_msh(str(head_mesh_path))
    tags = np.asarray(mesh.elm.tag1, dtype=int)
    scalp_mask = tags == scalp_tag
    if not np.any(scalp_mask):
        # Fallback: try a common alternate tag
        scalp_mask = tags == 5
    if not np.any(scalp_mask):
        raise ValueError(
            f"No elements found with scalp tag {scalp_tag} (or 5) in {head_mesh_path}"
        )

    # Get unique node indices belonging to scalp elements
    node_ids = set()
    elm_data = mesh.elm.node_number_list[scalp_mask]
    for row in elm_data:
        for nid in row:
            if nid > 0:
                node_ids.add(int(nid))

    node_indices = sorted(node_ids)
    coords = np.asarray(mesh.nodes.node_coord, dtype=float)
    # SimNIBS uses 1-based indexing
    scalp_points = coords[np.array(node_indices) - 1]
    return scalp_points


def helmet_to_scalp_projection(
    helmet_positions: np.ndarray,
    scalp_points: np.ndarray,
) -> np.ndarray:
    """Project helmet positions onto the nearest point on the scalp surface.

    Parameters
    ----------
    helmet_positions : ndarray of shape (N, 3)
    scalp_points : ndarray of shape (M, 3)

    Returns
    -------
    projected : ndarray of shape (N, 3)
        Scalp surface points closest to each helmet position.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(scalp_points)
    _, indices = tree.query(helmet_positions)
    return scalp_points[indices]
