#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

do_searches=1
do_dashboard=1
do_arch_smoke=1
traj_spec="${TRAJ_SPEC:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_v1_full.sh [options]

Options:
  --no-searches      Skip calibration + position-grid runs
  --no-dashboard     Skip dashboard refresh
  --no-arch-smoke    Skip TI/NTS/hybrid synthetic smoke + tests
  --traj-spec PATH   Trajectory spec JSON/YAML path for Stage 13 in smoke runs
  -h, --help         Show this help

Environment variables (common):
  SIMNIBS_PY, PYTHON_BIN, HEAD_MESH, COIL_FILE, LOG_ROOT

For search customization:
  SWEEP_DIDT, TARGET_E_VPM, CENTRES, YDIRS, DIDT_AS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-searches)
      do_searches=0
      ;;
    --no-dashboard)
      do_dashboard=0
      ;;
    --no-arch-smoke)
      do_arch_smoke=0
      ;;
    --traj-spec)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --traj-spec" >&2
        usage
        exit 1
      fi
      traj_spec="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ "${do_searches}" == "1" ]]; then
  "${SCRIPT_DIR}/run_long_searches.sh"
fi

if [[ "${do_dashboard}" == "1" ]]; then
  "${SCRIPT_DIR}/run_dashboard_refresh.sh"
fi

if [[ "${do_arch_smoke}" == "1" ]]; then
  if [[ -n "${traj_spec}" ]]; then
    TRAJ_SPEC="${traj_spec}" "${SCRIPT_DIR}/run_architecture_smoke.sh"
  else
    "${SCRIPT_DIR}/run_architecture_smoke.sh"
  fi
fi

echo "v1 full automation run complete."
