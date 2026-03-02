#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set +e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

CONFIG_FILE="${SCRIPT_DIR}/config/tests_js.yml"
CONFIG_READER="${SCRIPT_DIR}/lib/coverage_config.py"
SUMMARY_FILE="$(cov_summary_output_file)"
JS_TOTAL_EXECUTED=0
JS_SKIPPED=0
FAILED_JS_TESTS=()
SKIPPED_JS_TESTS=()

if [[ ! -f "${CONFIG_FILE}" ]]; then
    cov_error "JS tests config file is missing: ${CONFIG_FILE}"
    exit 1
fi

cd "${JS_DIR}" || exit 1
npm i
npm i --no-save c8

run_js_cmd() {
    local test_name="$1"
    local cmd="$2"

    JS_TOTAL_EXECUTED=$((JS_TOTAL_EXECUTED + 1))
    echo "========== Running JS test: ${test_name} =========="
    bash -lc "${cmd}"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        FAILED_JS_TESTS+=("${test_name} (exit ${rc})")
    fi
}

while IFS= read -r raw_line; do
    IFS='|' read -r test_name enabled skip_reason kind command <<< "${raw_line//$'\t'/|}"
    [[ -z "${test_name}" ]] && continue

    if [[ "${enabled}" != "1" ]]; then
        JS_SKIPPED=$((JS_SKIPPED + 1))
        SKIPPED_JS_TESTS+=("${test_name} (${skip_reason})")
        continue
    fi

    case "${kind}" in
        command)
            run_js_cmd "${test_name}" "${command}"
            ;;
        *)
            JS_SKIPPED=$((JS_SKIPPED + 1))
            SKIPPED_JS_TESTS+=("${test_name} (unknown kind: ${kind})")
            ;;
    esac
done < <(python3 "${CONFIG_READER}" --suite js --profile "${TEST_PROFILE}" --config "${CONFIG_FILE}")

if [[ -f "${OV_WORKSPACE}/js-coverage/lcov.info" ]]; then
    cp "${OV_WORKSPACE}/js-coverage/lcov.info" "${OV_WORKSPACE}/js-lcov.info"
fi

JS_FAILED=${#FAILED_JS_TESTS[@]}
JS_PASSED=$((JS_TOTAL_EXECUTED - JS_FAILED))
cov_set_env "JS_TESTS_TOTAL" "$((JS_TOTAL_EXECUTED + JS_SKIPPED))"
cov_set_env "JS_TESTS_PASSED" "${JS_PASSED}"
cov_set_env "JS_TESTS_FAILED" "${JS_FAILED}"
cov_set_env "JS_TESTS_SKIPPED" "${JS_SKIPPED}"

{
    echo ""
    echo "## JS coverage test execution summary"
    echo "JS tests executed: ${JS_TOTAL_EXECUTED}"
    echo "JS tests passed: ${JS_PASSED}"
    echo "JS tests failed: ${JS_FAILED}"
    echo "JS tests skipped: ${JS_SKIPPED}"
    echo ""
    if [[ ${#FAILED_JS_TESTS[@]} -gt 0 ]]; then
        echo "Failed tests:"
        for item in "${FAILED_JS_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Failed tests: none"
    fi
    echo ""
    if [[ ${#SKIPPED_JS_TESTS[@]} -gt 0 ]]; then
        echo "Skipped tests:"
        for item in "${SKIPPED_JS_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Skipped tests: none"
    fi
} >> "${SUMMARY_FILE}"

if [[ ${#FAILED_JS_TESTS[@]} -gt 0 ]]; then
    cov_warn "One or more JS tests failed; continuing to coverage generation."
fi

exit 0
