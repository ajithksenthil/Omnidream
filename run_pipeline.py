"""End-to-end Omnidream TMS array optimisation pipeline.

Chains all stages from config loading through GA optimisation to SAC
training and report generation.  Supports synthetic mode (no SimNIBS)
for rapid prototyping and testing.

Usage:
  python run_pipeline.py --mode TI --synthetic --generations 10
  python run_pipeline.py --config config.yaml --mode NTS --stages 1-8
  python run_pipeline.py --config config.yaml --mode hybrid --stages 1-10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import OmnidreamConfig, load_config, paper_baseline, save_config


def parse_stages(raw: str) -> set[int]:
    """Parse stage ranges like '1-8' or '1,3,5-7'."""
    stages: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            stages.update(range(int(a), int(b) + 1))
        elif part:
            stages.add(int(part))
    return stages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omnidream end-to-end pipeline")
    p.add_argument("--config", type=str, default="", help="Path to YAML config")
    p.add_argument("--mode", choices=["superposition", "TI", "NTS", "hybrid"], default="TI")
    p.add_argument("--stages", type=str, default="1-14", help="Which stages to run (e.g. '1-8')")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic basis matrix (no SimNIBS required)")
    p.add_argument("--num-coils", type=int, default=16)
    p.add_argument("--pop-size", type=int, default=10)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--sac-episodes", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="pipeline_output")
    p.add_argument(
        "--traj-spec",
        type=str,
        default="",
        help="Path to trajectory spec JSON/YAML for Stage 13 (optional).",
    )
    # Atlas (Stage 14) overrides
    p.add_argument("--atlas-k-neighbors", type=int, default=None,
                   help="Atlas k-neighbors (overrides config)")
    p.add_argument("--atlas-max-new", type=int, default=None,
                   help="Max new charts during densification")
    p.add_argument("--atlas-no-hessians", action="store_true",
                   help="Skip Hessian computation in atlas charts")
    return p.parse_args()


def load_trajectory_spec_dict(path: str) -> dict[str, Any]:
    """Load a trajectory spec dict from JSON/YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory spec file not found: {p}")
    ext = p.suffix.lower()
    raw = p.read_text(encoding="utf-8")
    if ext == ".json":
        return json.loads(raw)
    if ext in (".yml", ".yaml"):
        import yaml

        data = yaml.safe_load(raw)
        return data if isinstance(data, dict) else {}
    raise ValueError(f"Unsupported trajectory spec extension: {ext}")


def main() -> None:
    args = parse_args()
    stages = parse_stages(args.stages)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== Omnidream Pipeline ({args.mode} mode) ===")
    print(f"Stages: {sorted(stages)}")
    print(f"Synthetic: {args.synthetic}")
    print()

    # ------------------------------------------------------------------
    # Stage 1: Load config
    # ------------------------------------------------------------------
    if 1 in stages:
        print("[Stage 1] Loading configuration...")
        if args.config:
            config = load_config(args.config)
        else:
            config = paper_baseline()
        save_config(config, output / "config_used.yaml")
        print(f"  Config saved → {output / 'config_used.yaml'}")
    else:
        config = paper_baseline()

    # ------------------------------------------------------------------
    # Stage 2: Build/load coil model
    # ------------------------------------------------------------------
    if 2 in stages:
        print("[Stage 2] Coil model check...")
        coil_path = Path(__file__).resolve().parent / "coil_models" / "c_shaped_miniature_v1.tcd"
        if coil_path.exists():
            print(f"  Coil model found: {coil_path}")
        else:
            print("  Coil model not found — will build if SimNIBS available.")

    # ------------------------------------------------------------------
    # Stage 3: Generate helmet positions
    # ------------------------------------------------------------------
    if 3 in stages:
        print("[Stage 3] Generating helmet positions...")
        from helmet_geometry import sample_helmet_positions, compute_helmet_normals, enforce_min_distance_on_sphere

        positions = sample_helmet_positions(
            args.num_coils,
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
        np.save(output / "helmet_positions.npy", positions)
        np.save(output / "helmet_orientations.npy", orientations)
        print(f"  {args.num_coils} coils placed on helmet surface")
    else:
        positions = None
        orientations = None

    # ------------------------------------------------------------------
    # Stage 4: Compute or load basis fields
    # ------------------------------------------------------------------
    if 4 in stages:
        print("[Stage 4] Basis field computation...")
        if args.synthetic:
            from basis_fields import generate_synthetic_basis
            basis_matrix, target_idx, surface_indices = generate_synthetic_basis(
                n_coils=args.num_coils,
            )
            print(f"  Synthetic basis: {basis_matrix.shape}")
        else:
            from basis_fields import compute_all_basis_fields
            if positions is None:
                positions = np.load(output / "helmet_positions.npy")
                orientations = np.load(output / "helmet_orientations.npy")
            basis_matrix = compute_all_basis_fields(config, positions, orientations)
            target_idx = basis_matrix.shape[0] // 2  # placeholder
            surface_indices = np.arange(int(basis_matrix.shape[0] * 0.9), basis_matrix.shape[0])

        np.savez_compressed(
            output / "basis_data.npz",
            basis_matrix=basis_matrix,
            target_idx=np.array([target_idx]),
            surface_indices=surface_indices,
        )
        print(f"  Basis saved → {output / 'basis_data.npz'}")
    else:
        # Load from previous run
        data = np.load(output / "basis_data.npz")
        basis_matrix = data["basis_matrix"]
        target_idx = int(data["target_idx"][0])
        surface_indices = data["surface_indices"]

    # ------------------------------------------------------------------
    # Stage 5: Build inductance matrix
    # ------------------------------------------------------------------
    if 5 in stages:
        print("[Stage 5] Building inductance matrix...")
        from coupling import build_inductance_matrix, coupling_coefficient

        if positions is None:
            positions = np.load(output / "helmet_positions.npy")
        L_matrix = build_inductance_matrix(positions, config.coil)
        np.save(output / "inductance_matrix.npy", L_matrix)

        k_max = 0.0
        N = L_matrix.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                k = coupling_coefficient(L_matrix[i, j], L_matrix[i, i], L_matrix[j, j])
                k_max = max(k_max, k)
        print(f"  L matrix: {L_matrix.shape}, max coupling k={k_max:.6f}")

    # ------------------------------------------------------------------
    # Stage 6: Run GA optimisation
    # ------------------------------------------------------------------
    if 6 in stages:
        print(f"[Stage 6] Running GA ({args.mode} mode, {args.generations} gens, pop={args.pop_size})...")
        from optimal_configuration import run_genetic_algorithm

        best, score, history = run_genetic_algorithm(
            basis_matrix=basis_matrix,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            pop_size=args.pop_size,
            num_coils=args.num_coils,
            generations=args.generations,
            mode=args.mode,
        )

        ga_result = {
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
        }
        with (output / "ga_best.json").open("w") as f:
            json.dump(ga_result, f, indent=2)
        with (output / "ga_history.json").open("w") as f:
            json.dump(history, f, indent=2)
        print(f"  Best fitness: {score:.6f}")
        print(f"  Results → {output / 'ga_best.json'}")

    # ------------------------------------------------------------------
    # Stage 7: Validate top-K (placeholder for full SimNIBS)
    # ------------------------------------------------------------------
    if 7 in stages:
        print("[Stage 7] Tier-2 validation (skipped in synthetic mode)")

    # ------------------------------------------------------------------
    # Stage 8: Coupling-compensated voltage commands
    # ------------------------------------------------------------------
    if 8 in stages:
        print("[Stage 8] Computing coupling-compensated voltages...")
        from coupling import build_impedance_matrix, compensate_coupling

        if not (output / "inductance_matrix.npy").exists():
            print("  Skipping — no inductance matrix available.")
        else:
            L_matrix = np.load(output / "inductance_matrix.npy")
            with (output / "ga_best.json").open() as f:
                ga_result = json.load(f)

            I_desired = np.array(ga_result["amplitudes"])
            freq = ga_result.get("freq_carrier", config.ti.freq_carrier_hz)
            Z = build_impedance_matrix(L_matrix, config.coil.coil_resistance_ohm, freq)
            V_cmd = compensate_coupling(I_desired, Z)

            cmd_data = {
                "I_desired": I_desired.tolist(),
                "V_command_real": np.real(V_cmd).tolist(),
                "V_command_imag": np.imag(V_cmd).tolist(),
                "freq_hz": freq,
            }
            with (output / "voltage_commands.json").open("w") as f:
                json.dump(cmd_data, f, indent=2)
            print(f"  Voltage commands → {output / 'voltage_commands.json'}")

    # ------------------------------------------------------------------
    # Stage 9: Train SAC agent
    # ------------------------------------------------------------------
    if 9 in stages:
        print(f"[Stage 9] Training SAC agent ({args.sac_episodes} episodes)...")
        try:
            from sac_tms_control import BrainEnv_TI, BrainEnv_NTS, BrainEnv_Hybrid, train_sac
        except ImportError as e:
            print(f"  ⚠ Skipping SAC training: {e}")
            print("  Install PyTorch to enable SAC training: pip install torch")
            stages.discard(9)
    if 9 in stages:

        # Load GA result for group assignment
        group = None
        if (output / "ga_best.json").exists():
            with (output / "ga_best.json").open() as f:
                ga_result = json.load(f)
            group = np.array(ga_result["group"])

        if args.mode == "TI":
            env = BrainEnv_TI(basis_matrix, target_idx, surface_indices, group, config)
        elif args.mode == "NTS":
            env = BrainEnv_NTS(basis_matrix, target_idx, surface_indices, config=config)
        elif args.mode == "hybrid":
            env = BrainEnv_Hybrid(basis_matrix, target_idx, surface_indices, group, config)
        else:
            env = BrainEnv_TI(basis_matrix, target_idx, surface_indices, group, config)

        agent, sac_history = train_sac(env, episodes=args.sac_episodes)
        with (output / "sac_history.json").open("w") as f:
            json.dump(sac_history, f, indent=2, default=str)
        print(f"  SAC history → {output / 'sac_history.json'}")

    # ------------------------------------------------------------------
    # Stage 11: Sensitivity analysis
    # ------------------------------------------------------------------
    if 11 in stages:
        print("[Stage 11] Running sensitivity analysis...")
        from sensitivity import run_sensitivity_analysis, print_sensitivity_summary

        sens_results = run_sensitivity_analysis(
            basis_matrix=basis_matrix,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=args.mode if args.mode != "hybrid" else "hybrid",
            alpha_max=1.0,
            n_reachable_samples=500,
            n_pareto_weights=50,
        )
        summary_str = print_sensitivity_summary(sens_results)
        print(summary_str)

        # Save results (numpy arrays serialised via npz)
        save_dict = {}
        for k, v in sens_results.items():
            if isinstance(v, np.ndarray):
                save_dict[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        save_dict[f"{k}__{kk}"] = vv
        if save_dict:
            np.savez_compressed(output / "sensitivity_analysis.npz", **save_dict)
            print(f"  Results → {output / 'sensitivity_analysis.npz'}")

        # Also save the text summary
        with (output / "sensitivity_summary.txt").open("w") as f:
            f.write(summary_str)
        print(f"  Summary → {output / 'sensitivity_summary.txt'}")

        # Control framework analysis
        print("  Running plant analysis...")
        from control_framework import TMSPlant, StimulationMode, analyse_plant

        mode_map = {"TI": StimulationMode.TI, "NTS": StimulationMode.NTS,
                     "hybrid": StimulationMode.HYBRID}
        plant = TMSPlant(
            basis_matrix=basis_matrix,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=mode_map.get(args.mode, StimulationMode.TI),
            positions_mm=positions if positions is not None else None,
        )
        analysis = analyse_plant(plant)
        print(f"  Condition number (state):  κ = {analysis['kappa_state']:.2f}")
        print(f"  Condition number (params): κ = {analysis['kappa_params']:.2f}")
        print(f"  Controllability rank:      {analysis['controllability_rank']} / {analysis['n_state']}")
        print(f"  Observability rank:        {analysis['observability_rank']} / {analysis['n_state']}")
        print(f"  Output controllability:    {analysis['output_controllability_rank']} / {analysis['n_output']}")

    # ------------------------------------------------------------------
    # Stage 12: CP Bridge Analysis
    # ------------------------------------------------------------------
    if 12 in stages:
        print("[Stage 12] Running CP bridge analysis...")
        from cp_bridge import run_cp_bridge_analysis, print_cp_bridge_summary

        # Use Pareto results from sensitivity analysis if available
        pareto_results = None
        if 11 in stages and "sens_results" in dir():
            # Extract Pareto results for the current mode
            if args.mode.upper() in ("TI", "HYBRID") and "pareto_ti" in sens_results:
                pareto_results = sens_results["pareto_ti"]
            elif args.mode.upper() == "NTS" and "pareto_nts" in sens_results:
                pareto_results = sens_results["pareto_nts"]

        cp_results = run_cp_bridge_analysis(
            basis_matrix=basis_matrix,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=args.mode if args.mode != "hybrid" else "TI",
            pareto_results=pareto_results,
            positions_mm=positions if positions is not None else None,
        )

        cp_summary = print_cp_bridge_summary(cp_results)
        print(cp_summary)

        # Save results (numpy arrays + scalars)
        cp_save = {}
        for k, v in cp_results.items():
            if isinstance(v, np.ndarray):
                cp_save[k] = v
            elif isinstance(v, (int, float)):
                cp_save[k] = np.array([v])
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        cp_save[f"{k}__{kk}"] = vv
                    elif isinstance(vv, (int, float)):
                        cp_save[f"{k}__{kk}"] = np.array([vv])

        # Add world arrays if available
        if cp_results.get("worlds"):
            cp_save["world_energies"] = np.array([w.energy for w in cp_results["worlds"]])
            cp_save["world_coherences"] = np.array([w.phi for w in cp_results["worlds"]])
            cp_save["world_probabilities"] = np.array([w.probability for w in cp_results["worlds"]])
            cp_save["world_amplitudes"] = np.array([w.amplitudes for w in cp_results["worlds"]])
            cp_save["world_outputs"] = np.array([w.outputs for w in cp_results["worlds"]])

        # Add group info for plots
        N = basis_matrix.shape[1]
        group = np.zeros(N)
        group[N // 2:] = 1.0
        cp_save["group"] = group

        # Add MORL objectives
        if "morl_objectives" in cp_results:
            for mk, mv in cp_results["morl_objectives"].items():
                cp_save[f"morl__{mk}"] = np.array([mv])

        if cp_save:
            np.savez_compressed(output / "cp_bridge_analysis.npz", **cp_save)
            print(f"  Results → {output / 'cp_bridge_analysis.npz'}")

        with (output / "cp_bridge_summary.txt").open("w") as f:
            f.write(cp_summary)
        print(f"  Summary → {output / 'cp_bridge_summary.txt'}")

    # ------------------------------------------------------------------
    # Stage 13: Trajectory Planning
    # ------------------------------------------------------------------
    if 13 in stages:
        print("[Stage 13] Planning trajectory through world states...")
        from trajectory import (
            PropertyGoal, TrajectorySpec, plan_trajectory, print_trajectory_summary,
            save_trajectory,
        )
        from control_framework import TMSPlant, StimulationMode

        # Build plant if not already available
        if "plant" not in dir():
            mode_map = {"TI": StimulationMode.TI, "NTS": StimulationMode.NTS,
                         "hybrid": StimulationMode.HYBRID}
            plant = TMSPlant(
                basis_matrix=basis_matrix,
                target_idx=target_idx,
                surface_indices=surface_indices,
                config=config,
                mode=mode_map.get(args.mode, StimulationMode.TI),
                positions_mm=positions if positions is not None else None,
            )

        # Get worlds from Stage 12
        traj_worlds = []
        if 12 in stages and "cp_results" in dir() and cp_results.get("worlds"):
            traj_worlds = cp_results["worlds"]

        if not traj_worlds:
            print("  Skipping — no worlds available (run Stage 11 + Stage 12 first).")
        else:
            # Sort worlds by energy to support default trajectory and defaults
            sorted_ids = sorted(range(len(traj_worlds)), key=lambda i: traj_worlds[i].energy)
            n_w = len(sorted_ids)

            traj_spec = None
            if args.traj_spec:
                try:
                    spec_dict = load_trajectory_spec_dict(args.traj_spec)
                    raw_targets = spec_dict.get("property_targets")
                    if raw_targets is not None:
                        spec_dict["property_targets"] = [
                            PropertyGoal(**pt) for pt in raw_targets
                        ]
                    traj_spec = TrajectorySpec(**spec_dict)
                    print(f"  Loaded trajectory spec → {args.traj_spec}")
                except Exception as exc:
                    print(f"  Failed to load trajectory spec ({exc}); using default.")

            if traj_spec is None:
                if n_w < 3:
                    print("  Skipping — fewer than 3 worlds available for default trajectory.")
                    traj_spec = None
                else:
                    waypoint_ids = [sorted_ids[0], sorted_ids[n_w // 2], sorted_ids[-1]]
                    traj_spec = TrajectorySpec(
                        spec_type="world_sequence",
                        world_ids=waypoint_ids,
                        n_steps_per_segment=30,
                        interpolation="linear",
                        safety_mode="permissive",
                        h_theorem_enforce=False,
                        name=f"{args.mode}_energy_sweep",
                    )

            if traj_spec is not None:
                # Fill missing IDs for convenience when user picks world_sequence/energy_descent.
                if traj_spec.spec_type == "world_sequence" and not traj_spec.world_ids:
                    if n_w >= 3:
                        traj_spec.world_ids = [sorted_ids[0], sorted_ids[n_w // 2], sorted_ids[-1]]
                    else:
                        traj_spec.world_ids = [sorted_ids[0], sorted_ids[-1]]
                if traj_spec.spec_type == "energy_descent":
                    if traj_spec.start_world_id is None:
                        traj_spec.start_world_id = sorted_ids[0]
                    if traj_spec.end_world_id is None:
                        traj_spec.end_world_id = sorted_ids[-1]

                traj_result = plan_trajectory(
                    spec=traj_spec,
                    worlds=traj_worlds,
                    plant=plant,
                    basis_matrix=basis_matrix,
                    target_idx=target_idx,
                    surface_indices=surface_indices,
                    L_matrix=plant.L,
                    config=config,
                )

                traj_summary = print_trajectory_summary(traj_result)
                print(traj_summary)
                save_trajectory(traj_result, output)
                print(f"  Trajectory → {output / 'trajectory_result.npz'}")

    # ------------------------------------------------------------------
    # Stage 14: Perceptual Outcome Atlas
    # ------------------------------------------------------------------
    if 14 in stages:
        print("[Stage 14] Building perceptual outcome atlas...")
        from atlas import build_atlas, densify_atlas, print_atlas_summary, save_atlas
        from control_framework import TMSPlant, StimulationMode

        # Apply CLI overrides to atlas config
        if args.atlas_k_neighbors is not None:
            config.atlas.k_neighbors = args.atlas_k_neighbors
        if args.atlas_max_new is not None:
            config.atlas.max_new_charts = args.atlas_max_new
        if args.atlas_no_hessians:
            config.atlas.compute_hessians = False
        else:
            # Default: skip hessians in pipeline for speed
            config.atlas.compute_hessians = False

        # Build plant if not already available
        if "plant" not in dir():
            mode_map = {"TI": StimulationMode.TI, "NTS": StimulationMode.NTS,
                         "hybrid": StimulationMode.HYBRID}
            plant = TMSPlant(
                basis_matrix=basis_matrix,
                target_idx=target_idx,
                surface_indices=surface_indices,
                config=config,
                mode=mode_map.get(args.mode, StimulationMode.TI),
                positions_mm=positions if positions is not None else None,
            )

        # Get worlds from Stage 12
        atlas_worlds = []
        if 12 in stages and "cp_results" in dir() and cp_results.get("worlds"):
            atlas_worlds = cp_results["worlds"]

        if not atlas_worlds:
            print("  Skipping — no worlds available (run Stages 11 + 12 first).")
        else:
            import math as _math
            from cp_bridge import (
                CPBridgeConfig as _CPBridgeConfig,
                compute_transfer_entropy_matrix as _cte_matrix,
                compute_phases_from_groups as _cpfg,
            )

            _cp_cfg = _CPBridgeConfig()
            _L = plant.L
            _te = _cte_matrix(_L, _cp_cfg.noise_variance)

            N = basis_matrix.shape[1]
            _group = np.zeros(N)
            _group[N // 2:] = 1.0
            _f1 = config.ti.freq_carrier_hz
            _f2 = _f1 + config.ti.delta_freq_default_hz
            _phases = _cpfg(_group, _f1, _f2, t=0.0)
            _eta = np.where(_group > 0.5, 1.0, 0.1)

            atlas = build_atlas(
                worlds=atlas_worlds,
                plant=plant,
                te_matrix=_te,
                phases=_phases,
                basis_matrix=basis_matrix,
                target_idx=target_idx,
                eta_field=_eta,
                group=_group,
                config=config,
                cp_config=_cp_cfg,
                k_neighbors=config.atlas.k_neighbors,
                compute_hessians=config.atlas.compute_hessians,
                atlas_config=config.atlas,
            )

            _max_new = config.atlas.max_new_charts
            if args.atlas_max_new is None:
                # Default cap: min(config value, len/3)
                _max_new = min(_max_new, max(1, len(atlas_worlds) // 3))

            atlas = densify_atlas(
                atlas, plant,
                te_matrix=_te, phases=_phases,
                basis_matrix=basis_matrix, target_idx=target_idx,
                eta_field=_eta, group=_group, config=config,
                cp_config=_cp_cfg,
                max_new=_max_new,
                atlas_config=config.atlas,
            )

            atlas_summary = print_atlas_summary(atlas)
            print(atlas_summary)

            save_atlas(atlas, output / "atlas_result.npz")
            print(f"  Atlas → {output / 'atlas_result.npz'}")

            with (output / "atlas_summary.txt").open("w") as f:
                f.write(atlas_summary)

    # ------------------------------------------------------------------
    # Stage 10: Generate report
    # ------------------------------------------------------------------
    if 10 in stages:
        print("[Stage 10] Generating summary report...")
        elapsed = time.time() - t0
        report = {
            "mode": args.mode,
            "synthetic": args.synthetic,
            "num_coils": args.num_coils,
            "stages_run": sorted(stages),
            "elapsed_seconds": elapsed,
        }
        if (output / "ga_best.json").exists():
            with (output / "ga_best.json").open() as f:
                report["ga_result"] = json.load(f)
        with (output / "pipeline_summary.json").open("w") as f:
            json.dump(report, f, indent=2)
        print(f"  Summary → {output / 'pipeline_summary.json'}")

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
