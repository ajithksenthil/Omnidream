#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

CENTRE="${CENTRE:-C3}"
POS_YDIR="${POS_YDIR:-Cz}"
DIDT_AS="${DIDT_AS:-687479035.6513342}"
TARGET_E_VPM="${TARGET_E_VPM:-7.2}"
GM_TAG="${GM_TAG:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OMNIDREAM_DIR}/simulations/single_cases}"
SESSION_NAME="${SESSION_NAME:-single_${CENTRE//[^A-Za-z0-9]/_}_${POS_YDIR//[^A-Za-z0-9]/_}_$(timestamp)}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_single_simnibs_case.sh [options]

Options:
  --centre <label-or-xyz>      Coil centre (default: C3)
  --pos-ydir <label-or-xyz>    Coil y-direction target (default: Cz)
  --didt <float>               Absolute dI/dt in A/s (default: 6.874790356e8)
  --target-e-vpm <float>       Target E_max in V/m for error report (default: 7.2)
  --gm-tag <int>               Tissue tag for GM metrics (default: 2)
  --session-name <name>        Fixed session directory name
  --output-root <path>         Output root (default: simulations/single_cases)
  -h, --help                   Show this help

Environment overrides:
  SIMNIBS_PY, HEAD_MESH, COIL_FILE, LOG_ROOT
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --centre)
      CENTRE="$2"
      shift 2
      ;;
    --pos-ydir)
      POS_YDIR="$2"
      shift 2
      ;;
    --didt)
      DIDT_AS="$2"
      shift 2
      ;;
    --target-e-vpm)
      TARGET_E_VPM="$2"
      shift 2
      ;;
    --gm-tag)
      GM_TAG="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
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
done

mkdir -p "${OUTPUT_ROOT}"
require_file "${SIMNIBS_PY}"
require_file "${HEAD_MESH}"
require_file "${COIL_FILE}"

run_and_log "single_simnibs_case" \
  "${SIMNIBS_PY}" "${OMNIDREAM_DIR}/field_calculator.py" \
  --head-mesh "${HEAD_MESH}" \
  --coil-file "${COIL_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --session-name "${SESSION_NAME}" \
  --centre "${CENTRE}" \
  --pos-ydir "${POS_YDIR}" \
  --didt "${DIDT_AS}" \
  --target-e-vpm "${TARGET_E_VPM}" \
  --gm-tag "${GM_TAG}" \
  --expected-runs 1

SESSION_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"
SUMMARY_PATH="${SESSION_DIR}/calibration_summary.json"
CONCISE_PATH="${SESSION_DIR}/single_case_result.json"

echo "${SESSION_DIR}" > "${LOG_ROOT}/latest_single_case_session.txt"

"${PYTHON_BIN}" - <<'PY' "${SUMMARY_PATH}" "${TARGET_E_VPM}" "${CONCISE_PATH}" "${CENTRE}" "${POS_YDIR}"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
target_e = float(sys.argv[2])
out_path = Path(sys.argv[3])
centre = sys.argv[4]
pos_ydir = sys.argv[5]

obj = json.loads(summary_path.read_text(encoding="utf-8"))
rows = obj.get("rows", [])
if not rows:
    raise SystemExit("No run rows found in summary.")
row = rows[0]
e_max = float(row["e_max"])
abs_err = abs(e_max - target_e)
rel_err = 100.0 * abs_err / max(abs(target_e), 1e-12)

concise = {
    "session_name": obj.get("session_name"),
    "session_dir": str(summary_path.parent),
    "centre": centre,
    "pos_ydir": pos_ydir,
    "didt_as": float(row["didt_as"]),
    "target_e_vpm": target_e,
    "e_max_vpm": e_max,
    "abs_error_vpm": abs_err,
    "rel_error_percent": rel_err,
    "goal_hit_5pct": rel_err <= 5.0,
    "result_mesh": row.get("result_mesh"),
    "run_dir": row.get("run_dir"),
    "run_error": row.get("run_error"),
}

out_path.write_text(json.dumps(concise, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(concise, indent=2, sort_keys=True))
PY

echo "Session dir: ${SESSION_DIR}"
echo "Summary: ${SUMMARY_PATH}"
echo "Concise result: ${CONCISE_PATH}"

