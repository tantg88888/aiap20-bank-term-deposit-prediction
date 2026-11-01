#!/usr/bin/env bash
# Runs the end-to-end ML pipeline. Dependencies must already be installed
# (pip install -r requirements.txt); this script does not install anything.
set -euo pipefail

# Resolve the project root (the directory containing this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
PYTHON_BIN="${PIPELINE_PYTHON:-python3}"

echo "Project root: ${PROJECT_ROOT}"
echo "Running the ML pipeline (src/aiap_batch20.py) with $(${PYTHON_BIN} --version)..."

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/aiap_batch20.py"

echo "Pipeline execution finished successfully."