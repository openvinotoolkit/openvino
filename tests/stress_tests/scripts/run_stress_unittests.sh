#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Helper script to execute OpenVINO StressUnitTests with the recommended combined logging approach:
#  1. Structured XML test report (--gtest_output=xml:...)
#  2. Full raw console output streamed and saved to file (tee test_run.log)
#  3. Automated kernel dmesg and NPU firmware debug log collection on failure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRESS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configurations
BIN_PATH="${STRESS_UNIT_TESTS_BIN:-${STRESS_ROOT}/bin/intel64/Release/StressUnitTests}"
TEST_CONF=""
GTEST_FILTER=""
OUTPUT_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- EXTRA_GTEST_OR_STRESS_FLAGS]

Options:
  -c, --test_conf <path>       Path to stress test configuration XML file (required or passed via flags)
  -f, --gtest_filter <filter>  GoogleTest filter pattern (e.g. 'StressUnitTests/UnitTestSuite.stress_load_unload/*')
  -o, --output_dir <dir>       Directory to store XML reports, run logs, and failure dumps (default: ./test_results_<timestamp>)
  -b, --bin <path>             Path to StressUnitTests binary (default: ${BIN_PATH})
  -h, --help                   Display this help message

Environment Variables:
  LD_LIBRARY_PATH              Must contain paths to OpenVINO runtime, TBB, and NPU driver libraries.
  STRESS_UNIT_TESTS_BIN        Override path to StressUnitTests binary.

Examples:
  # Run load/unload stress tests with combined logging:
  $(basename "$0") --test_conf /path/to/test_config.xml --gtest_filter 'StressUnitTests/UnitTestSuite.stress_load_unload/*'

  # Custom output directory:
  $(basename "$0") -c /path/to/test_config.xml -f 'StressUnitTests/UnitTestSuite.stress_*' -o ./my_run_logs
EOF
    exit 1
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--test_conf)
            TEST_CONF="$2"
            shift 2
            ;;
        -f|--gtest_filter)
            GTEST_FILTER="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--bin)
            BIN_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "${BIN_PATH}" ]]; then
    # If not found directly, check if StressUnitTests is in PATH
    if command -v StressUnitTests &>/dev/null; then
        BIN_PATH="$(command -v StressUnitTests)"
    else
        echo "[ERROR] StressUnitTests binary not found or not executable at: ${BIN_PATH}"
        echo "Please specify binary path with --bin <path> or set STRESS_UNIT_TESTS_BIN."
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"
FAILURE_LOGS_DIR="${OUTPUT_DIR}/failure_logs"
XML_REPORT="${OUTPUT_DIR}/test_results.xml"
RUN_LOG="${OUTPUT_DIR}/test_run.log"

CMD=("${BIN_PATH}")

if [[ -n "${TEST_CONF}" ]]; then
    CMD+=("--test_conf=${TEST_CONF}")
fi

if [[ -n "${GTEST_FILTER}" ]]; then
    CMD+=("--gtest_filter=${GTEST_FILTER}")
fi

CMD+=("--gtest_output=xml:${XML_REPORT}")
CMD+=("--failure_logs_dir=${FAILURE_LOGS_DIR}")
CMD+=("--metrics_report_dir=${OUTPUT_DIR}")

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "========================================================================"
echo " Starting OpenVINO StressUnitTests Execution"
echo "========================================================================"
echo " Binary:          ${BIN_PATH}"
echo " Output Dir:      ${OUTPUT_DIR}"
echo " XML Report:      ${XML_REPORT}"
echo " Metrics Report:  ${OUTPUT_DIR}/stress_test_metrics_report.txt"
echo " Console Log:     ${RUN_LOG}"
echo " Failure Logs:    ${FAILURE_LOGS_DIR}"
echo " Command:         ${CMD[*]}"
echo "========================================================================"

# Execute command with live console streaming and capture to file preserving exit code
TEST_EXIT_CODE=0
"${CMD[@]}" 2>&1 | tee "${RUN_LOG}" || TEST_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo " Execution Finished (Exit Code: ${TEST_EXIT_CODE})"
echo "========================================================================"
echo " Artifacts:"
echo "   - Test Results XML: ${XML_REPORT}"
echo "   - Full Run Log:     ${RUN_LOG}"
echo "   - Metrics Report:   ${OUTPUT_DIR}/stress_test_metrics_report.txt"
echo "   - Metrics JSON:     ${OUTPUT_DIR}/stress_test_metrics_report.json"
if [[ -d "${FAILURE_LOGS_DIR}" && "$(ls -A "${FAILURE_LOGS_DIR}" 2>/dev/null)" ]]; then
    echo "   - Failure Reports:  ${FAILURE_LOGS_DIR}/ (Contains dmesg/fw_log dumps)"
fi
echo "========================================================================"

exit ${TEST_EXIT_CODE}
