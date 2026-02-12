#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${SCRIPT_DIR}/run_calibration_search.sh"
"${SCRIPT_DIR}/run_position_grid_search.sh"
"${SCRIPT_DIR}/run_dashboard_refresh.sh"

echo "Long searches complete."

