#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

POP_SIZE="${POP_SIZE:-10}"
GENERATIONS="${GENERATIONS:-10}"
SAC_EPISODES="${SAC_EPISODES:-5}"
NUM_COILS="${NUM_COILS:-16}"
PIPELINE_OUTPUT_BASE="${PIPELINE_OUTPUT_BASE:-${OMNIDREAM_DIR}/pipeline_output}"
TRAJ_SPEC="${TRAJ_SPEC:-${OMNIDREAM_DIR}/configs/trajectory_spec_example.json}"

traj_args=()
if [[ -n "${TRAJ_SPEC}" && -f "${TRAJ_SPEC}" ]]; then
  traj_args+=(--traj-spec "${TRAJ_SPEC}")
else
  echo "Warning: trajectory spec not found at '${TRAJ_SPEC}'. Running without --traj-spec."
fi

for mode in TI NTS hybrid; do
  out_dir="${PIPELINE_OUTPUT_BASE}/${mode}"
  mkdir -p "${out_dir}"
  run_and_log "pipeline_smoke_${mode}" \
    "${PYTHON_BIN}" "${OMNIDREAM_DIR}/run_pipeline.py" \
    --mode "${mode}" \
    --synthetic \
    --num-coils "${NUM_COILS}" \
    --pop-size "${POP_SIZE}" \
    --generations "${GENERATIONS}" \
    --sac-episodes "${SAC_EPISODES}" \
    --output-dir "${out_dir}" \
    "${traj_args[@]}"
done

run_and_log "tests_smoke" \
  "${SIMNIBS_PY}" -m pytest "${OMNIDREAM_DIR}/tests/test_all.py" -q

echo "Architecture smoke outputs: ${PIPELINE_OUTPUT_BASE}"
