# Omnidream Frontend Data Contract

This folder contains a TypeScript schema + loaders for React dashboards consuming
`pipeline_output/` artifacts.

## Files

- `omnidreamDataSchema.ts`  
  Types and runtime validation for:
  - `pipeline_summary.json`
  - `ga_best.json`
  - `voltage_commands.json`
  - `basis_data.json` (exported from `basis_data.npz`)
  - `sensitivity_analysis.json` (exported from `sensitivity_analysis.npz`)
  - `cp_bridge_analysis.json` (exported from `cp_bridge_analysis.npz`)
  - `trajectory_result.json` (exported from `trajectory_result.npz`)

- `omnidreamDataLoader.ts`  
  Loaders for browser and Node:
  - `loadOmnidreamBundleFromHttp(baseUrl, fileMapOverride?)`
  - `loadOmnidreamBundleFromFs(baseDir, fileMapOverride?)`

## Export NPZ Artifacts to JSON

Run from `Omnidream/`:

```bash
python3 scripts/export_pipeline_npz.py --pipeline-dir pipeline_output --pretty
```

This writes:

- `pipeline_output/web/basis_data.json`
- `pipeline_output/web/sensitivity_analysis.json`
- `pipeline_output/web/cp_bridge_analysis.json`
- `pipeline_output/web/trajectory_result.json`
- `pipeline_output/web/pipeline_npz_manifest.json`

## Minimal Usage (React)

```ts
import { loadOmnidreamBundleFromHttp } from "./omnidreamDataLoader";

const bundle = await loadOmnidreamBundleFromHttp("/pipeline_output");
console.log(bundle.trajectory_result.time.length);
console.log(bundle.cp_bridge_analysis.world_energies.length);
```

## Notes

- `trajectory_result.h_cog` is optional. Some runs include it, some do not.
- Validators enforce cross-file consistency (e.g. `num_coils`, row widths, and
  trajectory time-series lengths).
