# Final Build Constraints (v1)

## Purpose
This is the sign-off constraint set for building and validating the Omnidream miniature-coil system (simulation-first, hardware-ready).  
Any run or hardware profile outside these limits is out-of-scope for `v1`.

## Constraint Profile
- Machine-readable profile: `Omnidream/configs/final_build_v1.yaml`
- Python preset: `final_build_v1()` in `Omnidream/config.py`

## Scope Decisions (Finalized)
1. Baseline geometry and material profile follows paper-baseline values (`u_r = 75`, 30 turns, 5 mm base gap).
2. Helmet baseline uses 32 coils on a spherical cap (`r_inner = 90 mm`, min spacing `20 mm`).
3. Operational electrical limits are conservative (`max_current = 5 A`, `max_voltage = 60 V`).
4. TI continuous operation uses stricter thermal shutdown (`41 C critical`).
5. Calibrated target for single-coil benchmark remains `E_max = 7.2 V/m`.

## Hard Constraints (Must Pass)

| Domain | Constraint | Limit | Verification |
|---|---|---|---|
| Coil geometry | Total turns | `30` | `coil_models/*.summary.json` |
| Coil geometry | Winding footprint | `7 mm x 4 mm` | `coil_models/*.summary.json` + CAD check |
| Coil geometry | Base gap | `5.0 mm` | `coil_models/*.summary.json` |
| Coil geometry | Base axis angle | `75 deg` per side (`150 deg` included) | generator input + summary |
| Coil metadata | Max `dI/dt` metadata | `>= 1.0e9 A/s` | summary field `stimulator_max_didt_as` |
| Helmet | Coil count | `32` | config/profile check |
| Helmet | Min coil spacing | `>= 20 mm` | `helmet_geometry.enforce_min_distance_on_sphere` |
| Electrical | Per-coil current | `0..5 A` | safety monitor + runtime logs |
| Electrical | Per-coil voltage | `<= 60 V` operational cap | drive controller limits |
| Thermal pulsed | Warning/Critical | `40 C / 45 C` | safety monitor |
| Thermal continuous (TI) | Warning/Critical | `38 C / 41 C` | safety monitor |
| Duty cycle pulsed | Max | `5%` | schedule validator |
| TI beat frequency | `Delta f` range | `1..100 Hz` | optimizer/action bounds |
| NTS guard time | Min inter-pulse gap | `>= 200 us` | `enforce_guard_times()` |
| SAR proxy | Surface SAR | `<= 3.2 W/kg` | TI fitness/safety checks |
| Surface field | Per-pulse surface threshold | `<= 7.2 V/m` | NTS safety penalty check |

## Calibration and Simulation Acceptance

| Check | Pass Criteria | Artifact |
|---|---|---|
| Coil build | `.tcd` and `.summary.json` emitted | `coil_models/c_shaped_miniature_v1.*` |
| Calibration sweep | `calibration_summary.json` exists and contains valid runs | `simulations/calibration_*/**/calibration_summary.json` |
| Target calibration | At least one run with `|E_max - 7.2| / 7.2 <= 5%` | calibration summary rows |
| Position grid | `status.json` shows `progress_percent = 100` | `simulations/position_grid/*/status.json` |
| Shortlist | ranked shortlist emitted | `shortlist.json`, `shortlist.md` |
| Reliability | hard-failed runs in batch <= `5%` | events + status logs |

## Deep Targeting (TI/NTS/Hybrid) Constraints

| Domain | Constraint | Limit |
|---|---|---|
| TI carrier | Base frequency | `1000 Hz` default |
| TI beat | `Delta f` | `1..100 Hz` |
| NTS membrane constant | `tau_m` | `3 ms` |
| NTS integration window | `tau_window` | `5 ms` |
| GA amplitudes | bounded by safety current | `<= 5 A` |
| SAC action bounds | normalized action clipped and mapped to safety limits | enforced in env |

## Manufacturing and Integration Constraints (Build Prep)

| Item | Constraint |
|---|---|
| Wire | Copper, nominal `0.20 mm` conductor diameter |
| Core | Iron powder class matching `u_r = 75` baseline for v1 |
| Coil fixture tolerance | mechanical placement tolerance `<= 0.5 mm` |
| Helmet mount tolerance | coil center placement tolerance `<= 1.0 mm` |
| Temperature sensing | per-coil sensing with shutdown path, update >= `100 Hz` in continuous mode |
| Current sensing | per-coil current monitoring, warning at `4.5 A`, critical at `5.0 A` |
| Recording frontend | 16 channels, `20 kHz` sampling, 16-bit ADC baseline |

## Resolved Inconsistencies
1. Coil metadata `max dI/dt` is raised to `1.0e9 A/s` for v1 so calibrated operation (`~6.87479e8 A/s`) is within declared limits.
2. Operational voltage is fixed at `60 V` for v1, even if hardware can be designed with higher absolute ceiling.

## Release Checklist
1. Generate coil with v1 profile and confirm summary constraints.
2. Run calibration sweep and confirm target hit within tolerance.
3. Run position grid search and produce shortlist.
4. Generate dashboard outputs under `monitoring/`.
5. Run pipeline synthetic smoke for TI, NTS, hybrid.
6. Pass tests: `pytest Omnidream/tests/test_all.py`.
7. Freeze artifacts and config profile used for the run.

## Related Documents
- `Omnidream/Blueprints/system_state_diagram.md`
- `Omnidream/Blueprints/systematic_implementation_plan.md`
- `Omnidream/Blueprints/deep_targeting_formulations.md`
