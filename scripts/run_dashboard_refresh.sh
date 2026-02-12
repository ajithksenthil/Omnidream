#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SIM_ROOT="${SIM_ROOT:-${OMNIDREAM_DIR}/simulations}"
OUT_DIR="${OUT_DIR:-${OMNIDREAM_DIR}/monitoring}"
GOAL_E_VPM="${GOAL_E_VPM:-7.2}"
GOAL_TOLERANCE_FRAC="${GOAL_TOLERANCE_FRAC:-0.05}"
PIPELINE_DIR="${PIPELINE_DIR:-${OMNIDREAM_DIR}/pipeline_output}"

mkdir -p "${OUT_DIR}"

args=(
  "${OMNIDREAM_DIR}/progress_dashboard.py"
  --sim-root "${SIM_ROOT}"
  --out-dir "${OUT_DIR}"
  --goal-e-vpm "${GOAL_E_VPM}"
  --goal-tolerance-frac "${GOAL_TOLERANCE_FRAC}"
)

if [[ -d "${PIPELINE_DIR}" ]]; then
  args+=(--pipeline-dir "${PIPELINE_DIR}")
fi

run_and_log "dashboard_refresh" "${PYTHON_BIN}" "${args[@]}"

echo "Dashboard output directory: ${OUT_DIR}"
echo "Project status: ${OUT_DIR}/project_status.json"
echo "Position status: ${OUT_DIR}/position_status.json"

