import os
import sys
import numpy as np
import random
import datetime
from copy import deepcopy
import argparse
import multiprocessing
from multiprocessing import Pool
from functools import partial
import json

from simnibs import sim_struct, run_simnibs, mesh_io

# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEAD_MESH = os.path.join(BASE_DIR, 'simnibs4_examples', 'm2m_ernie', 'ernie.msh')

# Example coil path finder
try:
    from find_coil_path import get_coil_path
    COIL_FILE = get_coil_path()
except ImportError:
    # Fallback: adjust to a real coil file path
    COIL_FILE = '/path/to/your/coil_model/Magstim_70mm_Fig8.ccd'

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SIMULATIONS_DIR = os.path.join(BASE_DIR, 'simulations')
os.makedirs(SIMULATIONS_DIR, exist_ok=True)

#########################################
# Loading Realistic Target Patterns
#########################################
def load_targets():
    """
    Load target patterns from disk.
    Here we assume .npy files stored in `targets` folder, each representing a desired E-field pattern on the mesh.
    In practice, you might have 3D coordinates and need to match them to mesh nodes.
    """
    targets_dir = os.path.join(BASE_DIR, 'targets')
    # Example: load multiple target patterns
    target_files = [f for f in os.listdir(targets_dir) if f.endswith('.npy')]
    targets = [np.load(os.path.join(targets_dir, tf)) for tf in target_files]
    return targets

try:
    TARGETS = load_targets()
except:
    # Fallback if no targets available: use dummy random data
    print("Warning: Using dummy random targets.")
    TARGETS = [np.random.rand(5000) for _ in range(3)]

# Desired length of field vectors (assuming we match target vector sizes)
DESIRED_LENGTH = len(TARGETS[0]) if TARGETS else 5000


#########################################
# Physical Constraints for Coil Placement
#########################################
def load_scalp_surface():
    """
    Load the scalp surface points where coils can be placed.
    This should be a file (e.g., .npy) containing an Nx3 array of coordinates on the scalp surface.
    """
    scalp_file = os.path.join(BASE_DIR, 'models', 'scalp_points.npy')
    if os.path.isfile(scalp_file):
        scalp_points = np.load(scalp_file)
        return scalp_points
    else:
        # Fallback: create random points (not realistic)
        print("Warning: No scalp surface found, using random placement.")
        return np.random.uniform(-50, 50, size=(1000, 3))

SCALP_POINTS = load_scalp_surface()


#########################################
# Simulation Functions
#########################################
def create_simulation_session(coil_positions, coil_orientations, coil_intensities, output_dir):
    session = sim_struct.SESSION()
    session.fnamehead = HEAD_MESH
    session.pathfem = output_dir

    tmslist = session.add_tmslist()
    tmslist.fnamecoil = COIL_FILE

    for i in range(len(coil_intensities)):
        pos = tmslist.add_position()
        pos.centre = coil_positions[i, :].tolist()
        ydir = coil_positions[i, :] + coil_orientations[i, :]
        pos.pos_ydir = ydir.tolist()
        pos.didt = coil_intensities[i]

    return session

def run_field_simulation(coil_positions, coil_orientations, coil_intensities):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    sim_dir = os.path.join(SIMULATIONS_DIR, f"sim_{timestamp}")
    os.makedirs(sim_dir, exist_ok=True)

    session = create_simulation_session(coil_positions, coil_orientations, coil_intensities, sim_dir)
    run_simnibs(session)

    result_files = [f for f in os.listdir(sim_dir) if f.endswith('_scalar.msh')]
    if not result_files:
        raise FileNotFoundError("No result file found after simulation.")
    result_file = os.path.join(sim_dir, result_files[0])

    mesh = mesh_io.read_msh(result_file)
    if 'magnE' not in mesh.field:
        raise ValueError("No 'magnE' field found in the simulation results.")
    e_mag = mesh.field['magnE'].value

    # Resize or pad to DESIRED_LENGTH
    if len(e_mag) > DESIRED_LENGTH:
        e_mag = e_mag[:DESIRED_LENGTH]
    elif len(e_mag) < DESIRED_LENGTH:
        e_mag = np.pad(e_mag, (0, DESIRED_LENGTH - len(e_mag)), 'constant')

    return e_mag

#########################################
# Precompute Basis Fields for Each Coil
#########################################
def precompute_coil_fields(num_coils, reference_intensity=1.0):
    """
    Precompute the field from each coil individually at a reference intensity.
    We'll fix all but one coil to intensity 0, and one coil to `reference_intensity`.
    Store these fields for use in linear combination later.
    """
    # This assumes we have a global coil configuration to test. In the GA, coil placement changes.
    # So we must pass coil placement to this function. Instead, we can do it per individual when needed.
    pass

#########################################
# Fitness Function with Inverse Problem
#########################################
def solve_inverse_problem(basis_matrix, target):
    """
    Solve a least-squares problem to find coil intensities that best approximate the target.
    basis_matrix: shape (DESIRED_LENGTH, num_coils) fields from each coil
    target: shape (DESIRED_LENGTH,) desired field pattern
    returns: coil_intensities (num_coils,)
    """
    # x = argmin ||Ax - b||2
    # A = basis_matrix, b = target
    # np.linalg.lstsq
    intensities, residuals, rank, s = np.linalg.lstsq(basis_matrix, target, rcond=None)
    return intensities

def fitness_function(individual, targets):
    """
    More advanced fitness: 
    1. For each coil in the configuration, compute field at reference intensity.
    2. Form a basis matrix where each column is one coil's E-field.
    3. For each target, solve the inverse problem to find best coil intensities.
    4. Compute error (MSE).
    5. Average errors over all targets.
    """
    # Step 1: compute basis fields for each coil individually
    # num_coils
    num_coils = len(individual['intensities'])
    basis_fields = []
    for c in range(num_coils):
        coil_intensities = np.zeros(num_coils)
        coil_intensities[c] = 1.0  # reference intensity
        field = run_field_simulation(individual['positions'], individual['orientations'], coil_intensities)
        basis_fields.append(field)
    basis_matrix = np.column_stack(basis_fields)  # shape (DESIRED_LENGTH, num_coils)

    errors = []
    for t in targets:
        intensities = solve_inverse_problem(basis_matrix, t)
        # Compute error for this target with these intensities
        approx = basis_matrix @ intensities
        mse = np.mean((approx - t)**2)
        errors.append(mse)

    return np.mean(errors)


#########################################
# Population Initialization with Constraints
#########################################
def ensure_min_distance(positions, min_dist=10.0):
    """
    Ensure that coils are at least min_dist mm apart.
    If any coil is too close, reposition it until constraints are met or give up after attempts.
    """
    # Simple iterative approach:
    attempts = 0
    max_attempts = 100
    while True:
        distances = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :])**2, axis=2))
        np.fill_diagonal(distances, 1e9)
        if np.min(distances) < min_dist:
            # Find the offending coil and move it
            idx = np.unravel_index(np.argmin(distances), distances.shape)
            # Move the second coil to a new random position
            rand_idx = np.random.randint(len(SCALP_POINTS))
            positions[idx[1], :] = SCALP_POINTS[rand_idx, :]
            attempts += 1
            if attempts > max_attempts:
                break
        else:
            break
    return positions

def initialize_population(pop_size, num_coils):
    population = []
    for _ in range(pop_size):
        # Choose random points from scalp surface
        chosen_indices = np.random.choice(len(SCALP_POINTS), size=num_coils, replace=False)
        positions = SCALP_POINTS[chosen_indices, :].copy()

        # Ensure min distance
        positions = ensure_min_distance(positions, min_dist=10.0)

        orientations = np.random.uniform(-1, 1, size=(num_coils, 3))
        norms = np.linalg.norm(orientations, axis=1, keepdims=True)
        orientations = orientations / norms
        intensities = np.random.uniform(0.5, 1.0, size=num_coils)
        
        individual = {
            'positions': positions,
            'orientations': orientations,
            'intensities': intensities
        }
        population.append(individual)
    return population

def crossover(parent1, parent2):
    child = {}
    for key in ['positions', 'orientations', 'intensities']:
        crossover_point = np.random.randint(1, len(parent1[key]))
        child[key] = np.vstack([parent1[key][:crossover_point], parent2[key][crossover_point:]])
    return child

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual['positions'])):
        if random.random() < mutation_rate:
            # Move coil along scalp if available
            idx = np.random.randint(len(SCALP_POINTS))
            individual['positions'][i, :] = SCALP_POINTS[idx, :]
        if random.random() < mutation_rate:
            individual['orientations'][i] += np.random.normal(0, 0.1, size=3)
            individual['orientations'][i] /= np.linalg.norm(individual['orientations'][i])
        if random.random() < mutation_rate:
            individual['intensities'][i] = np.clip(
                individual['intensities'][i] + np.random.normal(0,0.05), 0.0, 2.0)

    # Ensure min distance again after mutation
    individual['positions'] = ensure_min_distance(individual['positions'], min_dist=10.0)


#########################################
# Parallel Evaluation
#########################################
def evaluate_population(population, targets):
    with Pool(processes=min(len(population), multiprocessing.cpu_count())) as p:
        fitnesses = p.map(partial(fitness_function, targets=targets), population)
    return fitnesses


#########################################
# Genetic Algorithm
#########################################
def run_genetic_algorithm(pop_size=10, num_coils=5, generations=5, elite_fraction=0.2):
    population = initialize_population(pop_size, num_coils)
    fitnesses = evaluate_population(population, TARGETS)

    for gen in range(generations):
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitnesses)
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        best_fitness = fitnesses[0]
        median_fitness = np.median(fitnesses)
        print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}, Median Fitness = {median_fitness:.4f}")

        # Save checkpoint
        gen_file = os.path.join(RESULTS_DIR, f"gen_{gen}.json")
        with open(gen_file, 'w') as f:
            json.dump({
                'generation': gen,
                'best_fitness': best_fitness,
                'median_fitness': median_fitness,
                'population': [
                    {
                        'positions': ind['positions'].tolist(),
                        'orientations': ind['orientations'].tolist(),
                        'intensities': ind['intensities'].tolist()
                    } for ind in population
                ],
                'fitnesses': fitnesses
            }, f, indent=4)

        # Elitism
        elite_count = int(elite_fraction * pop_size)
        new_population = deepcopy(population[:elite_count])

        # Create offspring
        while len(new_population) < pop_size:
            parents = random.sample(population[:elite_count], 2)
            child = crossover(parents[0], parents[1])
            mutate(child, mutation_rate=0.1)
            new_population.append(child)

        population = new_population
        fitnesses = evaluate_population(population, TARGETS)

    # Final sort
    sorted_indices = np.argsort(fitnesses)
    population = [population[i] for i in sorted_indices]
    fitnesses = [fitnesses[i] for i in sorted_indices]

    best_config = population[0]
    best_fitness = fitnesses[0]

    return best_config, best_fitness


#########################################
# Main
#########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm for Optimal TMS Configuration")
    parser.add_argument('--pop_size', type=int, default=10, help='Population size')
    parser.add_argument('--num_coils', type=int, default=5, help='Number of coils')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--elite_fraction', type=float, default=0.2, help='Fraction of elites to keep each generation')
    args = parser.parse_args()

    best_solution, best_score = run_genetic_algorithm(
        pop_size=args.pop_size,
        num_coils=args.num_coils,
        generations=args.generations,
        elite_fraction=args.elite_fraction
    )
    print("Optimization Complete")
    print("Best solution fitness:", best_score)
    print("Best coil configuration:", best_solution)
    sys.exit(0)