#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

SUMMARY_FILE="$(cov_summary_output_file)"

OVERALL_TOTAL=$(( ${CXX_TESTS_TOTAL:-0} + ${PY_TESTS_TOTAL:-0} + ${JS_TESTS_TOTAL:-0} ))
OVERALL_PASSED=$(( ${CXX_TESTS_PASSED:-0} + ${PY_TESTS_PASSED:-0} + ${JS_TESTS_PASSED:-0} ))
OVERALL_FAILED=$(( ${CXX_TESTS_FAILED:-0} + ${PY_TESTS_FAILED:-0} + ${JS_TESTS_FAILED:-0} ))
OVERALL_SKIPPED=$(( ${CXX_TESTS_SKIPPED:-0} + ${PY_TESTS_SKIPPED:-0} + ${JS_TESTS_SKIPPED:-0} ))

if [[ ${OVERALL_TOTAL} -gt 0 ]]; then
    OVERALL_PASS_RATE=$(awk "BEGIN {printf \"%.1f\", (${OVERALL_PASSED}*100)/${OVERALL_TOTAL}}")
else
    OVERALL_PASS_RATE="0.0"
fi

{
    echo "## Coverage Report Summary"
    echo ""
    echo "**Profile:** \`${TEST_PROFILE}\`  "
    echo "**Overall pass rate:** \`${OVERALL_PASS_RATE}%\`"
    echo ""
    echo "### Overall"
    echo "| Metric | Value |"
    echo "| --- | ---: |"
    echo "| Total test units | ${OVERALL_TOTAL} |"
    echo "| Passed | ${OVERALL_PASSED} |"
    echo "| Failed | ${OVERALL_FAILED} |"
    echo "| Skipped | ${OVERALL_SKIPPED} |"
    echo ""
    echo "### By Suite"
    echo "| Suite | Total | Passed | Failed | Skipped |"
    echo "| --- | ---: | ---: | ---: | ---: |"
    echo "| C++ | ${CXX_TESTS_TOTAL:-0} | ${CXX_TESTS_PASSED:-0} | ${CXX_TESTS_FAILED:-0} | ${CXX_TESTS_SKIPPED:-0} |"
    echo "| Python | ${PY_TESTS_TOTAL:-0} | ${PY_TESTS_PASSED:-0} | ${PY_TESTS_FAILED:-0} | ${PY_TESTS_SKIPPED:-0} |"
    echo "| JS | ${JS_TESTS_TOTAL:-0} | ${JS_TESTS_PASSED:-0} | ${JS_TESTS_FAILED:-0} | ${JS_TESTS_SKIPPED:-0} |"
    echo ""
    echo "### Coverage Files"
    echo "- C/C++ lcov: \`coverage.info\`"
    echo "- Python XML: \`python-coverage.xml\`"
    echo "- JS lcov: \`js-lcov.info\`"
    echo ""
    if [[ ! -f "${OV_WORKSPACE}/python-coverage.xml" ]]; then
        echo ":warning: Python coverage file is missing (\`python-coverage.xml\`)."
    fi
    if [[ ! -f "${OV_WORKSPACE}/js-lcov.info" ]]; then
        echo ":warning: JS coverage file is missing (\`js-lcov.info\`)."
    fi
    echo ""
    echo "### C/C++ Coverage Details (lcov)"
} >> "${SUMMARY_FILE}"

if [[ -f "${OV_WORKSPACE}/coverage.info" ]]; then
    {
        echo '```'
        lcov --summary "${OV_WORKSPACE}/coverage.info" || true
        echo '```'
    } >> "${SUMMARY_FILE}"
else
    echo "coverage.info not found (coverage generation likely failed earlier)." >> "${SUMMARY_FILE}"
fi
