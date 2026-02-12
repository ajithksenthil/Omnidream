#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SESSION_NAME="${SESSION_NAME:-auto_calibration_$(timestamp)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OMNIDREAM_DIR}/simulations/acceptance_calibration}"
SWEEP_DIDT="${SWEEP_DIDT:-4e8,6e8,6.874790356e8,7e8,8e8}"
TARGET_E_VPM="${TARGET_E_VPM:-7.2}"
CENTRE="${CENTRE:-C3}"
POS_YDIR="${POS_YDIR:-Cz}"
GM_TAG="${GM_TAG:-2}"

mkdir -p "${OUTPUT_ROOT}"
require_file "${SIMNIBS_PY}"
require_file "${HEAD_MESH}"
require_file "${COIL_FILE}"

run_and_log "calibration_search" \
  "${SIMNIBS_PY}" "${OMNIDREAM_DIR}/field_calculator.py" \
  --head-mesh "${HEAD_MESH}" \
  --coil-file "${COIL_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --session-name "${SESSION_NAME}" \
  --centre "${CENTRE}" \
  --pos-ydir "${POS_YDIR}" \
  --sweep-didt "${SWEEP_DIDT}" \
  --target-e-vpm "${TARGET_E_VPM}" \
  --gm-tag "${GM_TAG}"

SESSION_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"
SUMMARY_PATH="${SESSION_DIR}/calibration_summary.json"
LATEST_POINTER="${LOG_ROOT}/latest_calibration_session.txt"
echo "${SESSION_DIR}" > "${LATEST_POINTER}"

echo "Calibration session: ${SESSION_DIR}"
echo "Summary: ${SUMMARY_PATH}"

"${PYTHON_BIN}" - <<'PY' "${SUMMARY_PATH}" "${TARGET_E_VPM}"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
target_e = float(sys.argv[2])
obj = json.loads(summary_path.read_text(encoding="utf-8"))
rows = obj.get("rows", [])
if not rows:
    print("No calibration rows found.")
    raise SystemExit(0)

def abs_err(r):
    return abs(float(r["e_max"]) - target_e)

best = min(rows, key=abs_err)
rel = 100.0 * abs_err(best) / max(abs(target_e), 1e-12)
print("Best calibration point:")
print(f"  didt_as={best['didt_as']}")
print(f"  e_max={best['e_max']}")
print(f"  abs_error={abs_err(best)}")
print(f"  rel_error_percent={rel}")
PY

