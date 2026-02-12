# SimNIBS Miniature TMS Coil Implementation Plan

## Goal
Build a SimNIBS-based simulation pipeline for a miniature C-shaped TMS coil, grounded in the Dong Song lab paper and aligned with the existing `Omnidream` blueprints.

## System State Reference
- `Omnidream/Blueprints/system_state_diagram.md` documents the operational state machine, expected artifacts per state, and failure-recovery transitions for this pipeline.
- `Omnidream/Blueprints/systematic_implementation_plan.md` defines gate-based execution, timeline, metrics, and release criteria.
- `Omnidream/Blueprints/final_build_constraints.md` defines finalized numeric constraints and acceptance limits for v1 build.

## Source of truth for coil specs

Primary paper used:
- Wenxuan Jiang et al., *A C-shaped miniaturized coil for transcranial magnetic stimulation in rodents* (J. Neural Eng., 2023, DOI: 10.1088/1741-2552/acc097).

Paper-derived parameters to use first:
- Winding footprint at each base: `7 mm x 4 mm`
- Windings: `30 turns`
- Core geometry base values: toroid `OD 15.2 mm`, `ID 8.5 mm`, `height 6.0 mm`, modified to C-shape
- Base gap geometry: `150 deg` between bases, shortest base-to-base distance `5 mm`
- Core relative permeability used in FEM section: `u_r = 75`
- FEM winding simplification in paper: conductor diameter `0.2 mm`
- Calibration targets from paper:
  - Maximum `B` on cortex surface: `460 mT`
  - Maximum `E` on cortex surface: `7.2 V/m`
  - Protocol reference: `3 min at 10 Hz`

## Important discrepancy to resolve before coding

Current local blueprint assumptions conflict with the paper:
- `Omnidream/Blueprints/tms-coil-simulation.md` uses `u_r = 5000`, `core gap = 0.5 mm`, wire diameter `0.1 mm`.
- The paper's modeled core uses `u_r = 75`, with C-shape derived from a toroid and 5 mm base separation.

Decision for implementation:
- Treat the paper as baseline truth for v1.
- Run sensitivity sweeps afterward to test whether blueprint values were intended as an alternative design, not the replication target.

## Environment baseline (this machine)

- SimNIBS CLI detected at: `/Users/ajithsenthil/Applications/SimNIBS-4.5/bin/simnibs`
- Version: `4.5.0`
- Python env with SimNIBS: `/Users/ajithsenthil/Applications/SimNIBS-4.5/simnibs_env/bin/python`

## Phase plan

### Phase 0: Normalize model assumptions (0.5 day)
1. Create a single YAML/JSON config for coil and pulse parameters (`coil_config`).
2. Encode two parameter sets:
   - `paper_baseline`
   - `omnidream_variant`
3. Use SI units internally (meters, A/s, S/m, etc.) with explicit conversion helpers.

Acceptance:
- All scripts consume one shared config; no hardcoded geometry values remain.

### Phase 1: Build miniature coil model as SimNIBS `.tcd` (1-2 days)
1. Replace ad-hoc coil definition in `Omnidream/custom_c_shaped_coil.py` with SimNIBS 4.5 coil APIs:
   - `simnibs.simulation.tms_coil.tms_coil.TmsCoil`
   - `LineSegmentElements`
   - `TmsStimulator`
2. Generate a C-shaped wire path that matches `paper_baseline`.
3. Write coil to `Omnidream/coil_models/c_shaped_miniature_v1.tcd`.
4. Add a script to visualize and sanity-check:
   - winding continuity
   - winding count
   - gross dimensions

Acceptance:
- `.tcd` loads in a SimNIBS session without errors.
- Geometric checks pass against configured dimensions.

### Phase 2: Single-coil forward simulation and calibration (1-2 days)
1. Update `Omnidream/field_calculator.py` to:
   - auto-detect SimNIBS 4.5 resource path
   - use absolute `didt` values (A/s), not relative placeholder values
   - support custom `.tcd` coil input
2. Run controlled single-position simulations.
3. Calibrate `didt` so simulated field maxima are consistent with paper targets (`460 mT`, `7.2 V/m`) under matching geometry assumptions.
4. Save:
   - field maps
   - calibration curve (`didt -> maxE, maxB`)
   - run metadata (mesh, coil file hash, config hash)

Acceptance:
- Reproducible run reproduces paper-order magnitudes within agreed tolerance (for example +/-10%).

### Phase 3: Integrate with `Omnidream` optimization workflow (1-2 days)
1. In `Omnidream/optimal_configuration.py`:
   - switch default coil from `Magstim_70mm_Fig8.ccd` to configurable miniature `.tcd`
   - enforce physically plausible `didt` bounds from calibration curve
2. Add constraints:
   - coil spacing
   - orientation constraints
   - optional target ROI constraints
3. Ensure outputs include both optimization score and physical feasibility score.

Acceptance:
- GA runs end-to-end with custom miniature coil and emits ranked candidate layouts.

### Phase 4: Extend to multi-coil array from blueprint (2-3 days)
1. Implement `4x4` array geometry from local blueprint as a parameterized placement generator.
2. Start with superposition approximation for speed.
3. Re-run top candidates with full multi-position SimNIBS solves.
4. Quantify:
   - focality
   - interference index
   - ROI coverage

Acceptance:
- Top-N array layouts are reproducibly ranked with stored metrics and plots.

### Phase 5: Validation and reporting (1 day)
1. Produce a report artifact:
   - assumptions
   - paper-baseline replication metrics
   - divergence between `paper_baseline` and `omnidream_variant`
2. Add quick regression checks:
   - coil file validity
   - simulation run smoke test
   - field metric extraction test

Acceptance:
- One command reproduces key outputs from scratch on this workstation.

## Deliverables

1. `Omnidream/coil_models/c_shaped_miniature_v1.tcd`
2. Updated scripts:
   - `Omnidream/custom_c_shaped_coil.py`
   - `Omnidream/field_calculator.py`
   - `Omnidream/optimal_configuration.py`
3. Reproducible run config and metadata logs.
4. Validation report with calibration and optimization results.

## Risks and mitigations

- Risk: Human head mesh vs rodent setup mismatch.
  - Mitigation: clearly separate "paper replication mode" and "human translational mode"; do not mix metrics.

- Risk: Coil geometry under-specified for exact replication.
  - Mitigation: bracket with sensitivity sweeps (gap, winding diameter, core permeability).

- Risk: Thermal behavior not captured in pure SimNIBS EM runs.
  - Mitigation: keep thermal checks as external post-processing or COMSOL cross-check track.

## Immediate next actions

1. Lock `paper_baseline` parameter JSON/YAML.
2. Implement `.tcd` coil generator script for C-shape in SimNIBS 4.5 API.
3. Patch `field_calculator.py` to use that coil and absolute `didt` calibration sweeps.
