# DAI Controller Integration Guide (Omnidream)

This document maps nonlinear control ideas to the actual Omnidream code surface.
It is implementation-facing: where to plug in, what to compute, and what to test.

## Scope

- Plant and physics interfaces: `control_framework.py`
- Safety limits and operating bounds: `config.py` (`SafetyConfig`)
- Existing optimization/control loops: `optimal_configuration.py`, `sac_tms_control.py`
- CP/DAI bridge outputs for high-level objective shaping: `cp_bridge.py`

## 1) Current Control Stack in Repo

### Plant side (already implemented)

- `TMSPlant.forward_from_params(amplitudes, **kwargs) -> np.ndarray(5,)`
  - Returns `[target_metric, surface_metric, membrane_metric, SAR_max, T_max]`
- `TMSPlant.jacobian_output_wrt_amplitudes(amplitudes, **kwargs) -> np.ndarray(5, N)`
  - Finite-difference local sensitivity in parameter space
- `build_cost_function(plant, goal) -> cost_fn(amplitudes, **kwargs)`
  - Includes soft penalties for SAR and thermal violations
- `analyse_plant(...)`
  - Provides Jacobians, SVD spectrum, condition numbers, and linearization summary

### Controller side (already implemented)

- `Controller` ABC in `control_framework.py`
- `GreedyJacobianController`
  - Baseline one-step local policy using `J^T w`

### Optimizer side (already implemented)

- GA-based global search (`optimal_configuration.py`)
- Optional SAC environments (`sac_tms_control.py`)
- Trajectory/atlas layers for outcome-space planning (`trajectory.py`, `atlas.py`)

## 2) What the New Nonlinear Guidance Changes

The project already has nonlinear plant behavior and local Jacobian control, but it is missing a
single, explicit nonlinear closed-loop controller that enforces constraints over a finite horizon.

Implementation implication:

- Keep GA and SAC as global search tools.
- Add a horizon-based nonlinear controller for online steering and constraint handling.
- Use CP metrics as optional objective terms, not as hard dynamic constraints initially.

## 3) Recommended Controller to Add Next

### Nonlinear MPC (priority 1)

Reason:

- Directly fits existing interfaces (`forward_from_params`, safety limits, cost function).
- Handles SAR/temperature/current/voltage constraints explicitly.
- Provides deterministic, debuggable behavior for staged deployment.

Minimal formulation at step `k`:

- Decision: amplitude sequence `alpha[k:k+H-1]`
- Objective:
  - maximize target metric
  - minimize surface metric and control effort
  - optional regularizer for CP energy terms
- Constraints (each horizon step):
  - `SAR <= sar_limit_wkg`
  - `T <= temp_critical_continuous_c` (or pulsed threshold by mode)
  - `|I| <= max_current_a`
  - `|V| <= max_voltage_v`

Apply first action only, then re-solve (receding horizon).

### Sliding-mode or backstepping (priority 2)

Use only after NMPC baseline is stable. These need tighter assumptions about model form and can
introduce chattering or implementation complexity.

## 4) Exact Integration Points

1. New module:
   - `nonlinear_controller.py`
   - Add `NonlinearMPCController(Controller)` class
2. Reuse existing primitives:
   - `TMSPlant.forward_from_params`
   - `build_cost_function`
   - `map_output_vector`
3. Config extension (optional, recommended):
   - Add `ControllerConfig` in `config.py`
   - Fields: horizon, dt, lambda_slew, lambda_power, solver_maxiter
4. Pipeline hook:
   - Add optional execution path after Stage 11 (sensitivity) or Stage 13 (trajectory)
   - Persist output to `pipeline_output/nmpc_schedule.npz`

## 5) Safety Contract for Controller

All controller-generated schedules must satisfy:

- electrical:
  - `|I_i(t)| <= max_current_a`
  - `|V_i(t)| <= max_voltage_v`
- bio-safety:
  - `SAR_max(t) <= sar_limit_wkg`
  - `T_max(t) <= temp_critical_*`
- smoothness:
  - bounded slew `|alpha_i(t+1) - alpha_i(t)|` to avoid abrupt transients

If solver cannot find a feasible step:

- fallback to last feasible action
- emit explicit `infeasible_step` event in logs

## 6) Test Plan (Implementation Gate)

Add tests before enabling by default:

- unit:
  - controller returns bounded actions
  - infeasible cases trigger fallback path
- integration:
  - synthetic TI/NTS/hybrid smoke with NMPC enabled
  - no safety violations on produced schedule
- regression:
  - existing `tests/test_all.py` remains green
  - SAC-skipped environments remain unaffected when `torch` missing

## 7) Deliverable Definition

A "done" nonlinear controller integration means:

1. `NonlinearMPCController` merged and callable
2. Pipeline flag to run it (`--controller nmpc`)
3. Saved artifacts:
   - `nmpc_schedule.npz` (amplitudes over time)
   - `nmpc_metrics.json` (safety and objective traces)
4. Dashboard plots:
   - target/surface/SAR/T over time
   - feasibility status timeline

## 8) Immediate Next Task

Implement `nonlinear_controller.py` with a minimal receding-horizon solver using the existing
plant API, then add one synthetic smoke test plus one safety regression test.
