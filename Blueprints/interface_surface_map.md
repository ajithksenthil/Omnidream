# Interface Surface Map (Control -> DAI/MORL/4D-NCA)

## Scope
This document maps the concrete interface surface across:
- `control_framework.py`
- `sensitivity.py`
- `em_theory.py`
- `sac_tms_control.py`
- `optimal_configuration.py`

and shows how these connect to the bridge layer:
- `cp_bridge.py`
- `run_pipeline.py` stages `11` and `12`.

## 1) Class and Function Hierarchy

### `control_framework.py`
- `PlantModel(ABC)`:
  - `dims() -> PlantDimensions`
  - `forward(x, u) -> y`
  - `forward_from_params(amplitudes, **kwargs) -> y`
  - `linearize(x0, u0) -> (A, B, C, D)`
  - `jacobian_output_wrt_input(x, u) -> J`
  - plus controllability/observability helpers
- `TMSPlant(PlantModel)`:
  - state dimension: `2N + 2`
  - input dimension: `N`
  - output dimension: `5`
  - outputs: `[target_metric, surface_max, V_m_peak, SAR_max, T_max]`
  - modes: `TI`, `NTS`, `HYBRID`
  - parameter-space jacobian: `jacobian_output_wrt_amplitudes(...)`
- `Controller(ABC)`:
  - `select_action(state, goal, plant) -> u`
- `GreedyJacobianController(Controller)`:
  - one-step Jacobian ascent/descent policy
- `build_cost_function(...)`:
  - physics-aware scalar objective with SAR/T constraints
- `analyse_plant(...)`:
  - one-stop diagnostic package (Jacobians, ranks, condition numbers)

### `sensitivity.py`
- Jacobians:
  - `compute_jacobian_ti(...)`
  - `compute_jacobian_nts(...)`
  - `compute_jacobian_analytical_nts(...)`
- Curvature:
  - `compute_hessian(...)`
- Conditioning/SVD:
  - `compute_condition_number(...)`
  - `compute_hessian_condition(...)`
  - `svd_analysis(...)`
- Reachability:
  - `compute_reachable_set_ti(...)`
  - `compute_reachable_set_nts(...)`
- Pareto:
  - `compute_pareto_front_ti(...)`
  - `compute_pareto_front_nts(...)`
- Orchestration:
  - `run_sensitivity_analysis(...)`
  - `print_sensitivity_summary(...)`

### `em_theory.py`
- `MaxwellBasis`: quasi-static validity + superposition residual
- `TissueModel`: Cole-Cole conductivity (frequency-dependent)
- `CoilInductance`: self/mutual inductance + coupling + core loss
- `NeuralResponse`: membrane LPF and TI carrier/beat transmission
- `SARThermal`: SAR + steady-state/transient thermal approximations

### `sac_tms_control.py`
- environments:
  - `BrainEnv_TI` (action dim `N+1`)
  - `BrainEnv_NTS` (action dim `2N`)
  - `BrainEnv_Hybrid` (action dim `2N+1`)
- RL core:
  - `ReplayBuffer`, `MLP`, `GaussianPolicy`, `SACAgent`, `train_sac(...)`

### `optimal_configuration.py`
- genome:
  - `positions`, `orientations`, `amplitudes`, `group`, `fire_times`, `freq_carrier`, `delta_freq`, `mode`
- fitness dispatch:
  - superposition / TI / NTS / hybrid
- GA loop:
  - `initialize_population(...)`
  - `crossover(...)`
  - `mutate(...)`
  - `evaluate_population(...)`
  - `run_genetic_algorithm(...)`

## 2) Core Data Contracts

### Plant state/input/output (`control_framework.TMSPlant`)
- state `x` shape: `(2N + 2,)`
  - `x[:N]`: coil currents `I`
  - `x[N]`: membrane proxy `V_m`
  - `x[N+1:2N+1]`: coil temperatures
  - `x[2N+1]`: target metric cache
- input `u` shape: `(N,)`
  - per-coil voltage command
- output `y` shape: `(5,)`
  - `y[0]`: target objective (`M_target` TI or `V_target` NTS)
  - `y[1]`: surface max
  - `y[2]`: membrane peak proxy
  - `y[3]`: SAR max
  - `y[4]`: temperature max

### SAC env observation contract
- all envs expose:
  - `state_dim = N + 5`
  - observation includes previous amplitudes + target/surface scalars + 3-slot placeholder
- this is the simplest insertion point for DAI latent-state augmentation.

### GA-to-RL handoff contract
- GA emits:
  - `amplitudes`, `group`, `fire_times`, frequencies
- SAC consumes:
  - basis matrix + target/surface indices + optional fixed group
- practical bridge:
  - GA provides initialization/prior for SAC policies.

## 3) Mathematical Quantities Computed Today

### TI branch
- `A1, A2` group amplitudes
- modulation depth `M = 2*min(|A1|, |A2|)`
- target and surface statistics from `M`
- SAR from TI RMS field approximation

### NTS branch
- membrane-integration peak `V_peak`
- target and surface maxima from `V_peak`
- per-pulse surface safety max
- duty-cycle-adjusted SAR proxy

### Control-theory branch
- local linearization `(A,B,C,D)`
- Jacobians in state/input and parameter space
- controllability / observability ranks
- conditioning (`kappa`) and singular spectrum

### Multi-objective geometry branch
- reachable sets (sampled action-image geometry)
- Pareto fronts for depth-vs-surface trade-off
- dominated/non-dominated partitions

## 4) Existing DAI/MORL/4D-NCA Bridge

This repository already contains a direct mapping layer:
- `cp_bridge.py`:
  - coil-level agent states (`AgentState`, MB1)
  - group-level state (`GroupState`, MB2)
  - world-level Pareto embeddings (`WorldState`)
  - transfer entropy from mutual inductance
  - collective Phi, group free energy, MORL scalarization
  - 4D-NCA total energy decomposition

Pipeline integration already exists:
- `run_pipeline.py`:
  - stage `11`: sensitivity + plant analysis
  - stage `12`: CP bridge analysis (`run_cp_bridge_analysis`)

## 5) Steering Architecture (Recommended)

### Layer A: Physical plant layer
- keep `TMSPlant` as the authoritative forward model interface.

### Layer B: Sensitivity/geometry layer
- use `run_sensitivity_analysis` for local/global action geometry.
- use this to generate priors over controllable subspaces.

### Layer C: Policy layer
- use GA (`optimal_configuration.py`) for coarse structural search.
- use SAC (`sac_tms_control.py`) for online fine control.

### Layer D: DAI/MORL/NCA layer
- use `cp_bridge.py` energy terms as top-level steering objective.
- use world probabilities from Pareto fronts as policy priors.
- optimize expected utility under both safety and Phi/architecture terms.

## 6) Concrete Bridge Hooks

1. Hook `cp_bridge.scalarize_morl(...)` into SAC reward shaping.
2. Feed `analyse_plant(...)[\"J_params\"]` into adaptive exploration scaling.
3. Use `sensitivity.py` Pareto points to initialize GA population and SAC replay.
4. Add DAI latent state slots in env observation (replace 3-slot placeholder).
5. Use `cp_bridge.detect_implacement(...)` as scheduling signal for mode-switch TI/NTS/hybrid.

## 7) Interface Gaps to Resolve Next

1. Standardize objective naming across modules:
   - TI target metric currently represented as `M_target`
   - NTS target metric as `V_target`
   - unify schema in reporting/bridge adapters.
2. Introduce shared typed result payloads (dataclasses) instead of ad hoc dicts.
3. Promote explicit units in all logs and JSON fields (A, V, V/m, W/kg, C).
4. Add invariant checks for shape compatibility at module boundaries.

## 8) Minimal Execution Path

1. Run stages `1-11` to get basis, GA outputs, sensitivity geometry.
2. Run stage `12` to compute CP bridge metrics and world mapping.
3. Use bridge outputs to re-weight SAC/GA objectives and iterate.

Command:

```bash
python3 run_pipeline.py --mode hybrid --synthetic --stages 1-12
```

