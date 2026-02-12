#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

SESSION_NAME="${SESSION_NAME:-auto_grid_$(timestamp)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OMNIDREAM_DIR}/simulations/position_grid}"
CENTRES="${CENTRES:-C3,C4,F3,F4,P3,P4}"
YDIRS="${YDIRS:-Cz,Fz,Pz}"
TARGET_E_VPM="${TARGET_E_VPM:-7.2}"
TOLERANCE_FRAC="${TOLERANCE_FRAC:-0.05}"
TOP_K="${TOP_K:-10}"
GM_TAG="${GM_TAG:-2}"
AUTO_DIDT_FROM_CALIBRATION="${AUTO_DIDT_FROM_CALIBRATION:-1}"
DEFAULT_DIDT="${DEFAULT_DIDT:-687479035.6513342}"

mkdir -p "${OUTPUT_ROOT}"
require_file "${SIMNIBS_PY}"
require_file "${HEAD_MESH}"
require_file "${COIL_FILE}"

DIDT_AS="${DIDT_AS:-}"
if [[ -z "${DIDT_AS}" ]]; then
  DIDT_AS="${DEFAULT_DIDT}"
  latest_calibration_file="${LOG_ROOT}/latest_calibration_session.txt"
  if [[ "${AUTO_DIDT_FROM_CALIBRATION}" == "1" && -f "${latest_calibration_file}" ]]; then
    latest_calib_dir="$(cat "${latest_calibration_file}")"
    summary_path="${latest_calib_dir}/calibration_summary.json"
    if [[ -f "${summary_path}" ]]; then
      derived="$("${PYTHON_BIN}" - <<'PY' "${summary_path}" "${TARGET_E_VPM}"
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = float(sys.argv[2])
v = summary.get("suggested_didt_for_target_e_as")
if v is not None:
    print(v)
    raise SystemExit(0)
rows = summary.get("rows", [])
if not rows:
    raise SystemExit(0)
best = min(rows, key=lambda r: abs(float(r["e_max"]) - target))
print(best["didt_as"])
PY
)"
      if [[ -n "${derived}" ]]; then
        DIDT_AS="${derived}"
      fi
    fi
  fi
fi

echo "Using dI/dt: ${DIDT_AS} A/s"

run_and_log "position_grid_search" \
  "${SIMNIBS_PY}" "${OMNIDREAM_DIR}/position_grid_search.py" \
  --head-mesh "${HEAD_MESH}" \
  --coil-file "${COIL_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --session-name "${SESSION_NAME}" \
  --centres "${CENTRES}" \
  --ydirs "${YDIRS}" \
  --didt "${DIDT_AS}" \
  --target-e-vpm "${TARGET_E_VPM}" \
  --tolerance-frac "${TOLERANCE_FRAC}" \
  --gm-tag "${GM_TAG}" \
  --top-k "${TOP_K}"

SESSION_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"
GRID_RESULTS="${SESSION_DIR}/grid_results.json"
SHORTLIST="${SESSION_DIR}/shortlist.json"
LATEST_POINTER="${LOG_ROOT}/latest_grid_session.txt"
echo "${SESSION_DIR}" > "${LATEST_POINTER}"

echo "Grid session: ${SESSION_DIR}"
echo "Grid results: ${GRID_RESULTS}"
echo "Shortlist: ${SHORTLIST}"

"${PYTHON_BIN}" - <<'PY' "${SHORTLIST}" "${TOP_K}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
top_k = int(sys.argv[2])
obj = json.loads(path.read_text(encoding="utf-8"))
rows = obj.get("shortlist", [])
print("Top candidates:")
for idx, row in enumerate(rows[:top_k], start=1):
    print(
        f"  {idx}. {row['position_key']} | "
        f"E_max={row['e_max']:.6g} | "
        f"abs_error={row['abs_error_vpm']:.6g} V/m | "
        f"rel_error={row['rel_error_percent']:.3f}%"
    )
PY

