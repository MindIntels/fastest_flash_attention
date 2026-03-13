#!/usr/bin/env bash
# ==============================================================================
# Fastest Flash Attention — Test Runner
# ==============================================================================
# Usage:
#   bash run_tests.sh          # run all tests
#   bash run_tests.sh -k fa4   # run only FA4 tests
#   bash run_tests.sh -x       # stop on first failure
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure the project root is on PYTHONPATH so `fastest_flash_attention` is importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo " Fastest Flash Attention — Test Suite"
echo "=============================================="
echo "Project root : ${PROJECT_ROOT}"
echo "Test dir     : ${SCRIPT_DIR}/tests"
echo "Python       : $(python3 --version 2>&1)"
echo "PyTorch      : $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "=============================================="

python3 -m pytest "${SCRIPT_DIR}/tests/" -v --tb=short "$@"

echo ""
echo "All tests passed!"
