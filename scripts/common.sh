#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIDREAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${OMNIDREAM_DIR}/.." && pwd)"

SIMNIBS_PY_DEFAULT="/Users/ajithsenthil/Applications/SimNIBS-4.5/simnibs_env/bin/python3"
HEAD_MESH_DEFAULT="/Users/ajithsenthil/Desktop/Universum/OmniDream/TMS_Sim/tms_grid_project/simnibs4_examples/m2m_ernie/ernie.msh"

SIMNIBS_PY="${SIMNIBS_PY:-${SIMNIBS_PY_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HEAD_MESH="${HEAD_MESH:-${HEAD_MESH_DEFAULT}}"
COIL_FILE="${COIL_FILE:-${OMNIDREAM_DIR}/coil_models/c_shaped_miniature_v1.tcd}"
LOG_ROOT="${LOG_ROOT:-${OMNIDREAM_DIR}/results/automation_logs}"

mkdir -p "${LOG_ROOT}"

timestamp() {
  date "+%Y%m%d_%H%M%S"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

run_and_log() {
  local label="$1"
  shift
  local log_file="${LOG_ROOT}/${label}_$(timestamp).log"
  echo "[${label}] Log: ${log_file}"
  "$@" 2>&1 | tee "${log_file}"
}

