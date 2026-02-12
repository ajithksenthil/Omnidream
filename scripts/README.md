# Automation Scripts

These scripts automate the long-running SimNIBS searches and architecture smoke checks so implementation work can continue in parallel.

## Quick Start
From `Omnidream/`:

```bash
chmod +x scripts/*.sh
scripts/run_v1_full.sh
```

## Script Index
- `scripts/run_calibration_search.sh`  
  Runs `field_calculator.py` sweep for target calibration.

- `scripts/run_position_grid_search.sh`  
  Runs `position_grid_search.py` for `(centre, pos_ydir)` combinations.  
  If available, auto-loads `dI/dt` from the latest calibration session.

- `scripts/run_single_simnibs_case.sh`  
  Runs one SimNIBS scenario for a specific `(centre, pos_ydir, dI/dt)` and writes a concise JSON result.

- `scripts/run_dashboard_refresh.sh`  
  Rebuilds `monitoring/` status and plots from `simulations/`.

- `scripts/run_architecture_smoke.sh`  
  Runs TI/NTS/hybrid synthetic pipeline smoke runs + test gate.  
  Automatically passes `--traj-spec` to Stage 13 if `configs/trajectory_spec_example.json` exists
  (or if `TRAJ_SPEC` is set).

- `scripts/run_long_searches.sh`  
  Runs calibration + position-grid + dashboard in one command.

- `scripts/run_v1_full.sh`  
  Top-level orchestrator for long searches, dashboard, and architecture smoke.

- `scripts/export_pipeline_npz.py`
  Converts `pipeline_output/*.npz` artifacts into `pipeline_output/web/*.json`
  for frontend dashboards.

## Useful Environment Variables
- `SIMNIBS_PY`  
  SimNIBS Python interpreter path.

- `HEAD_MESH`  
  Head mesh path (`.msh`) for calibration and position search.

- `COIL_FILE`  
  Coil file path (`.tcd`).

- `TARGET_E_VPM`, `SWEEP_DIDT`, `CENTRES`, `YDIRS`, `DIDT_AS`  
  Search tuning parameters.

- `LOG_ROOT`  
  Log output root directory (default: `results/automation_logs`).

- `TRAJ_SPEC`  
  Optional path to trajectory spec JSON/YAML passed to `run_pipeline.py --traj-spec` during
  architecture smoke runs.  
  Default: `configs/trajectory_spec_example.json`.

## Trajectory Spec Example
- `configs/trajectory_spec_example.json`  
  Default Stage 13 trajectory spec used by automation scripts.  
  You can override it with:

```bash
TRAJ_SPEC=/absolute/path/to/my_spec.json scripts/run_architecture_smoke.sh
```

or:

```bash
scripts/run_v1_full.sh --traj-spec /absolute/path/to/my_spec.yaml
```

## Notes
- Logs are written to `results/automation_logs/*.log`.
- Latest calibration and grid session paths are recorded in:
  - `results/automation_logs/latest_calibration_session.txt`
  - `results/automation_logs/latest_grid_session.txt`
