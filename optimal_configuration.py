"""Genetic Algorithm for optimal TMS coil array configuration.

Supports four optimisation modes:
  - ``superposition``: classical weighted superposition (original)
  - ``TI``:            Temporal Interference modulation depth
  - ``NTS``:           Neural Temporal Summation peak V_m
  - ``hybrid``:        combined TI + NTS objective

The individual genome encodes positions, orientations, amplitudes,
group assignments (TI), firing times (NTS), and frequency parameters.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from config import OmnidreamConfig, load_config, paper_baseline
from helmet_geometry import (
    compute_helmet_normals,
    enforce_min_distance_on_sphere,
    sample_helmet_positions,
)
from ti_fields import ti_fitness
from nts_timing import nts_fitness, enforce_guard_times

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Individual representation
# ---------------------------------------------------------------------------

def create_individual(
    positions: np.ndarray,
    orientations: np.ndarray,
    amplitudes: np.ndarray,
    group: np.ndarray | None = None,
    fire_times: np.ndarray | None = None,
    freq_carrier: float = 1000.0,
    delta_freq: float = 10.0,
    mode: str = "superposition",
) -> dict:
    """Create a GA individual with all genes."""
    N = len(amplitudes)
    if group is None:
        group = np.random.randint(0, 2, size=N).astype(float)
    if fire_times is None:
        fire_times = np.sort(np.random.uniform(0, 5e-3, size=N))
    return {
        "positions": np.asarray(positions, dtype=float),
        "orientations": np.asarray(orientations, dtype=float),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "group": np.asarray(group, dtype=float),
        "fire_times": np.asarray(fire_times, dtype=float),
        "freq_carrier": float(freq_carrier),
        "delta_freq": float(delta_freq),
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------

def fitness_superposition(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    targets: list[np.ndarray] | None = None,
) -> float:
    """Original superposition fitness: MSE of least-squares reconstruction.

    If *targets* are provided, solves the inverse problem for each target
    and returns the average MSE.  Otherwise uses a simple focality metric.
    """
    amplitudes = individual["amplitudes"]

    if targets is not None and len(targets) > 0:
        errors = []
        for t in targets:
            # Solve: basis_matrix @ x ≈ t
            x, *_ = np.linalg.lstsq(basis_matrix, t, rcond=None)
            approx = basis_matrix @ x
            mse = float(np.mean((approx - t) ** 2))
            errors.append(mse)
        return float(np.mean(errors))

    # Focality metric: E at target vs mean surface E
    E = basis_matrix @ amplitudes
    E_target = float(np.mean(np.abs(E[target_idx])))
    E_surface = float(np.mean(np.abs(E[surface_indices])))
    # Lower is better → negate focality
    return -(E_target / max(E_surface, 1e-12))


def fitness_TI(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig | None = None,
) -> float:
    """TI modulation-depth fitness (lower is better)."""
    if config is None:
        config = paper_baseline()
    return ti_fitness(
        amplitudes=individual["amplitudes"],
        group=individual["group"],
        freq_carrier=individual["freq_carrier"],
        delta_freq=individual["delta_freq"],
        basis_matrix=basis_matrix,
        target_idx=target_idx,
        surface_indices=surface_indices,
        ti_cfg=config.ti,
        safety_cfg=config.safety,
    )


def fitness_NTS(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig | None = None,
) -> float:
    """NTS peak membrane potential fitness (lower is better)."""
    if config is None:
        config = paper_baseline()
    return nts_fitness(
        amplitudes=individual["amplitudes"],
        fire_times=individual["fire_times"],
        basis_matrix=basis_matrix,
        target_idx=target_idx,
        surface_indices=surface_indices,
        nts_cfg=config.nts,
        safety_cfg=config.safety,
    )


def fitness_hybrid(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig | None = None,
    w_ti: float = 0.6,
    w_nts: float = 0.4,
) -> float:
    """Hybrid TI + NTS fitness (lower is better)."""
    return w_ti * fitness_TI(individual, basis_matrix, target_idx, surface_indices, config) + \
           w_nts * fitness_NTS(individual, basis_matrix, target_idx, surface_indices, config)


def fitness_dispatch(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig | None = None,
    targets: list[np.ndarray] | None = None,
) -> float:
    """Route to the correct fitness function based on individual['mode']."""
    mode = individual.get("mode", "superposition")
    if mode == "superposition":
        return fitness_superposition(individual, basis_matrix, target_idx, surface_indices, targets)
    elif mode == "TI":
        return fitness_TI(individual, basis_matrix, target_idx, surface_indices, config)
    elif mode == "NTS":
        return fitness_NTS(individual, basis_matrix, target_idx, surface_indices, config)
    elif mode == "hybrid":
        return fitness_hybrid(individual, basis_matrix, target_idx, surface_indices, config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def initialize_population(
    pop_size: int,
    num_coils: int,
    mode: str = "superposition",
    config: OmnidreamConfig | None = None,
) -> list[dict]:
    """Create initial population on helmet surface."""
    if config is None:
        config = paper_baseline()

    population = []
    for _ in range(pop_size):
        positions = sample_helmet_positions(
            num_coils,
            r_inner=config.helmet.r_inner_mm,
            theta_max_deg=config.helmet.theta_max_deg,
        )
        positions = enforce_min_distance_on_sphere(
            positions,
            min_dist=config.helmet.min_spacing_mm,
            r_inner=config.helmet.r_inner_mm,
            theta_max_deg=config.helmet.theta_max_deg,
        )
        orientations = compute_helmet_normals(positions)
        amplitudes = np.random.uniform(0.5, 2.0, size=num_coils)
        group = np.random.randint(0, 2, size=num_coils).astype(float)
        fire_times = np.sort(np.random.uniform(0, config.nts.tau_window_s, size=num_coils))
        freq_carrier = config.ti.freq_carrier_hz
        delta_freq = config.ti.delta_freq_default_hz

        individual = create_individual(
            positions=positions,
            orientations=orientations,
            amplitudes=amplitudes,
            group=group,
            fire_times=fire_times,
            freq_carrier=freq_carrier,
            delta_freq=delta_freq,
            mode=mode,
        )
        population.append(individual)
    return population


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(parent1: dict, parent2: dict) -> dict:
    """Single-point crossover for array genes; arithmetic for scalars."""
    N = len(parent1["amplitudes"])
    cp = np.random.randint(1, N)

    child = {
        "positions": np.vstack([parent1["positions"][:cp], parent2["positions"][cp:]]),
        "orientations": np.vstack([parent1["orientations"][:cp], parent2["orientations"][cp:]]),
        "amplitudes": np.concatenate([parent1["amplitudes"][:cp], parent2["amplitudes"][cp:]]),
        "group": np.concatenate([parent1["group"][:cp], parent2["group"][cp:]]),
        "fire_times": np.concatenate([parent1["fire_times"][:cp], parent2["fire_times"][cp:]]),
        "freq_carrier": 0.5 * (parent1["freq_carrier"] + parent2["freq_carrier"]),
        "delta_freq": 0.5 * (parent1["delta_freq"] + parent2["delta_freq"]),
        "mode": parent1["mode"],
    }
    return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate(
    individual: dict,
    mutation_rate: float = 0.1,
    config: OmnidreamConfig | None = None,
) -> None:
    """Mutate an individual in place."""
    if config is None:
        config = paper_baseline()

    N = len(individual["amplitudes"])

    for i in range(N):
        # Position mutation: random re-sample on helmet
        if random.random() < mutation_rate:
            new_pos = sample_helmet_positions(
                1,
                r_inner=config.helmet.r_inner_mm,
                theta_max_deg=config.helmet.theta_max_deg,
            )
            individual["positions"][i] = new_pos[0]

        # Orientation mutation: Gaussian perturbation + renormalise
        if random.random() < mutation_rate:
            individual["orientations"][i] += np.random.normal(0, 0.1, size=3)
            norm = np.linalg.norm(individual["orientations"][i])
            if norm > 1e-8:
                individual["orientations"][i] /= norm

        # Amplitude mutation
        if random.random() < mutation_rate:
            individual["amplitudes"][i] = np.clip(
                individual["amplitudes"][i] + np.random.normal(0, 0.2),
                0.0, config.safety.max_current_a,
            )

        # Group mutation: bit flip
        if random.random() < mutation_rate:
            individual["group"][i] = 1.0 - individual["group"][i]

        # Fire time mutation
        if random.random() < mutation_rate:
            individual["fire_times"][i] = np.clip(
                individual["fire_times"][i] + np.random.normal(0, config.nts.tau_window_s * 0.1),
                0.0, config.nts.tau_window_s,
            )

    # Frequency mutations
    if random.random() < mutation_rate:
        individual["freq_carrier"] = max(
            500.0,
            individual["freq_carrier"] + np.random.normal(0, 200.0),
        )
    if random.random() < mutation_rate:
        individual["delta_freq"] = np.clip(
            individual["delta_freq"] + np.random.normal(0, 5.0),
            config.ti.delta_freq_min_hz,
            config.ti.delta_freq_max_hz,
        )

    # Enforce minimum coil distance
    individual["positions"] = enforce_min_distance_on_sphere(
        individual["positions"],
        min_dist=config.helmet.min_spacing_mm,
        r_inner=config.helmet.r_inner_mm,
        theta_max_deg=config.helmet.theta_max_deg,
    )

    # Enforce guard times for NTS
    if individual["mode"] in ("NTS", "hybrid"):
        individual["fire_times"] = enforce_guard_times(
            individual["fire_times"], config.nts.t_guard_s,
        )


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    individual: dict,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig,
    targets: list[np.ndarray] | None,
) -> float:
    return fitness_dispatch(individual, basis_matrix, target_idx, surface_indices, config, targets)


def evaluate_population(
    population: list[dict],
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig,
    targets: list[np.ndarray] | None = None,
    parallel: bool = False,
) -> list[float]:
    """Evaluate fitness for every individual."""
    if parallel and len(population) > 1:
        import multiprocessing
        with Pool(processes=min(len(population), multiprocessing.cpu_count())) as p:
            fn = partial(
                _evaluate_one,
                basis_matrix=basis_matrix,
                target_idx=target_idx,
                surface_indices=surface_indices,
                config=config,
                targets=targets,
            )
            return list(p.map(fn, population))
    else:
        return [
            fitness_dispatch(ind, basis_matrix, target_idx, surface_indices, config, targets)
            for ind in population
        ]


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    config: OmnidreamConfig | None = None,
    pop_size: int = 20,
    num_coils: int = 32,
    generations: int = 50,
    elite_fraction: float = 0.2,
    mode: str = "TI",
    targets: list[np.ndarray] | None = None,
    parallel: bool = False,
    verbose: bool = True,
) -> tuple[dict, float, list[dict]]:
    """Run the GA and return (best_individual, best_fitness, history).

    Parameters
    ----------
    basis_matrix : ndarray (num_points, N)
        Precomputed basis E-fields.
    target_idx : int or ndarray
        Index / indices of the target location in the basis matrix.
    surface_indices : ndarray
        Indices of cortical-surface sample points.
    config : OmnidreamConfig, optional
    pop_size : int
    num_coils : int
    generations : int
    elite_fraction : float
    mode : str
        One of ``superposition``, ``TI``, ``NTS``, ``hybrid``.
    targets : list of ndarray, optional
        Target E-field patterns for superposition mode.
    parallel : bool
    verbose : bool

    Returns
    -------
    best_individual : dict
    best_fitness : float
    history : list of dict  (per-generation stats)
    """
    if config is None:
        config = paper_baseline()

    population = initialize_population(pop_size, num_coils, mode, config)
    fitnesses = evaluate_population(
        population, basis_matrix, target_idx, surface_indices, config, targets, parallel,
    )

    history: list[dict] = []

    for gen in range(generations):
        # Sort by fitness (lower is better)
        sorted_idx = np.argsort(fitnesses)
        population = [population[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]

        best_f = fitnesses[0]
        median_f = float(np.median(fitnesses))
        if verbose:
            print(f"Gen {gen:3d}: best={best_f:.6f}  median={median_f:.6f}")

        history.append({
            "generation": gen,
            "best_fitness": best_f,
            "median_fitness": median_f,
        })

        # Save checkpoint
        ckpt = RESULTS_DIR / f"gen_{gen:04d}.json"
        with ckpt.open("w") as f:
            json.dump({
                "generation": gen,
                "best_fitness": best_f,
                "median_fitness": median_f,
                "mode": mode,
            }, f, indent=2)

        # Elitism
        elite_count = max(1, int(elite_fraction * pop_size))
        new_pop = deepcopy(population[:elite_count])

        # Offspring
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:elite_count], 2)
            child = crossover(p1, p2)
            mutate(child, mutation_rate=0.1, config=config)
            new_pop.append(child)

        population = new_pop
        fitnesses = evaluate_population(
            population, basis_matrix, target_idx, surface_indices, config, targets, parallel,
        )

    # Final sort
    sorted_idx = np.argsort(fitnesses)
    population = [population[i] for i in sorted_idx]
    fitnesses = [fitnesses[i] for i in sorted_idx]

    return population[0], fitnesses[0], history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omnidream GA optimiser")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")
    parser.add_argument("--mode", choices=["superposition", "TI", "NTS", "hybrid"], default="TI")
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--num-coils", type=int, default=32)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--target-coords", type=str, default="0,0,60",
                        help="Target x,y,z in mm (comma-separated)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic basis matrix (no SimNIBS required)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = paper_baseline()

    if args.synthetic:
        from basis_fields import generate_synthetic_basis
        basis_matrix, target_idx, surface_indices = generate_synthetic_basis(
            n_coils=args.num_coils,
        )
        print(f"Using synthetic basis: {basis_matrix.shape}")
    else:
        # Load real basis matrix
        from basis_fields import compute_all_basis_fields
        from helmet_geometry import sample_helmet_positions, compute_helmet_normals
        positions = sample_helmet_positions(args.num_coils, config.helmet.r_inner_mm, config.helmet.theta_max_deg)
        orientations = compute_helmet_normals(positions)
        basis_matrix = compute_all_basis_fields(config, positions, orientations)
        # Parse target coords
        target_coords = np.array([float(x) for x in args.target_coords.split(",")])
        from basis_fields import get_sample_point_indices
        target_idx, surface_indices = get_sample_point_indices(
            config.sim.head_mesh_path, target_coords, config.sim.gm_tag,
        )

    best, score, history = run_genetic_algorithm(
        basis_matrix=basis_matrix,
        target_idx=target_idx,
        surface_indices=surface_indices,
        config=config,
        pop_size=args.pop_size,
        num_coils=args.num_coils,
        generations=args.generations,
        elite_fraction=args.elite_fraction,
        mode=args.mode,
    )

    print(f"\nOptimisation complete ({args.mode} mode)")
    print(f"Best fitness: {score:.6f}")
    print(f"Results saved to: {RESULTS_DIR}")

    # Save final result
    final_path = RESULTS_DIR / "best_individual.json"
    with final_path.open("w") as f:
        json.dump({
            "mode": args.mode,
            "fitness": score,
            "num_coils": args.num_coils,
            "generations": args.generations,
            "positions": best["positions"].tolist(),
            "orientations": best["orientations"].tolist(),
            "amplitudes": best["amplitudes"].tolist(),
            "group": best["group"].tolist(),
            "fire_times": best["fire_times"].tolist(),
            "freq_carrier": best["freq_carrier"],
            "delta_freq": best["delta_freq"],
        }, f, indent=2)
    print(f"Best individual saved → {final_path}")


if __name__ == "__main__":
    main()
