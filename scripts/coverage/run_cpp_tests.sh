#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set +e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

CONFIG_FILE="${SCRIPT_DIR}/config/tests_cpp.yml"
CONFIG_READER="${SCRIPT_DIR}/lib/coverage_config.py"
FAILED_FILE="$(mktemp)"
SKIPPED_FILE="$(mktemp)"
EXECUTED_FILE="$(mktemp)"
SUMMARY_FILE="$(cov_summary_output_file)"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    cov_error "C++ tests config file is missing: ${CONFIG_FILE}"
    exit 1
fi

if [[ ! -f "${CONFIG_READER}" ]]; then
    cov_error "Coverage config reader is missing: ${CONFIG_READER}"
    exit 1
fi

# Prevent stale counters from previous runs contaminating current coverage.
if [[ -d "${BUILD_DIR}" ]]; then
    find "${BUILD_DIR}" -name '*.gcda' -delete || true
fi
if [[ -d "${BUILD_JS_DIR}" ]]; then
    find "${BUILD_JS_DIR}" -name '*.gcda' -delete || true
fi
rm -rf "${BUILD_DIR}/gcov" || true

run_test() {
    local test_name="$1"
    local binary_name="$2"
    local mode="${3:-gtest_single}"
    local test_args="${4:-}"
    local extra_env="${5:-}"

    local exe_path="${BIN_DIR}/${binary_name}"
    local rc

    if [[ ! -x "${exe_path}" ]]; then
        echo "${test_name} (missing binary: ${binary_name})" >> "${SKIPPED_FILE}"
        return 0
    fi

    echo "${test_name}" >> "${EXECUTED_FILE}"
    echo "========== Running ${test_name} =========="

    if [[ "${mode}" == "raw" ]]; then
        if [[ -n "${test_args}" ]]; then
            read -r -a raw_args <<< "${test_args}"
            if [[ -n "${extra_env}" ]]; then
                read -r -a extra_env_parts <<< "${extra_env}"
                env "${extra_env_parts[@]}" "${exe_path}" "${raw_args[@]}"
            else
                "${exe_path}" "${raw_args[@]}"
            fi
        else
            if [[ -n "${extra_env}" ]]; then
                read -r -a extra_env_parts <<< "${extra_env}"
                env "${extra_env_parts[@]}" "${exe_path}"
            else
                "${exe_path}"
            fi
        fi
    else
        if [[ -n "${test_args}" ]]; then
            if [[ -n "${extra_env}" ]]; then
                read -r -a extra_env_parts <<< "${extra_env}"
                env "${extra_env_parts[@]}" "${exe_path}" --gtest_filter="${test_args}"
            else
                "${exe_path}" --gtest_filter="${test_args}"
            fi
        else
            if [[ -n "${extra_env}" ]]; then
                read -r -a extra_env_parts <<< "${extra_env}"
                env "${extra_env_parts[@]}" "${exe_path}"
            else
                "${exe_path}"
            fi
        fi
    fi

    rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "${test_name} (exit ${rc})" >> "${FAILED_FILE}"
    fi

    return 0
}

resolve_args() {
    local args="$1"
    args="${args//__MODEL_PATH__/${MODEL_PATH}}"
    printf '%s\n' "${args}"
}

while IFS= read -r raw_line; do
    IFS='|' read -r test_name enabled skip_reason binary mode args extra_env <<< "${raw_line//$'\t'/|}"
    [[ -z "${test_name}" ]] && continue

    if [[ "${enabled}" != "1" ]]; then
        echo "${test_name} (${skip_reason})" >> "${SKIPPED_FILE}"
        continue
    fi

    args="$(resolve_args "${args}")"
    run_test "${test_name}" "${binary}" "${mode}" "${args}" "${extra_env}"
done < <(python3 "${CONFIG_READER}" --suite cpp --profile "${TEST_PROFILE}" --config "${CONFIG_FILE}")

mapfile -t FAILED_TESTS < "${FAILED_FILE}"
mapfile -t SKIPPED_TESTS < "${SKIPPED_FILE}"
mapfile -t EXECUTED_TESTS < "${EXECUTED_FILE}"
rm -f "${FAILED_FILE}" "${SKIPPED_FILE}" "${EXECUTED_FILE}"

CXX_TOTAL_EXECUTED=${#EXECUTED_TESTS[@]}
CXX_FAILED=${#FAILED_TESTS[@]}
CXX_SKIPPED=${#SKIPPED_TESTS[@]}
CXX_PASSED=$((CXX_TOTAL_EXECUTED - CXX_FAILED))
CXX_TOTAL_PLANNED=$((CXX_TOTAL_EXECUTED + CXX_SKIPPED))

cov_set_env "CXX_TESTS_TOTAL" "${CXX_TOTAL_PLANNED}"
cov_set_env "CXX_TESTS_EXECUTED" "${CXX_TOTAL_EXECUTED}"
cov_set_env "CXX_TESTS_PASSED" "${CXX_PASSED}"
cov_set_env "CXX_TESTS_FAILED" "${CXX_FAILED}"
cov_set_env "CXX_TESTS_SKIPPED" "${CXX_SKIPPED}"

{
    echo "## C++ coverage test execution summary"
    echo ""
    echo "Test profile: ${TEST_PROFILE}"
    echo ""
    echo "GPU mode: ${RUN_GPU_TESTS}"
    echo ""
    echo "NPU mode: ${RUN_NPU_TESTS}"
    echo ""
    echo "C++ tests planned: ${CXX_TOTAL_PLANNED}"
    echo "C++ tests executed: ${CXX_TOTAL_EXECUTED}"
    echo "C++ tests passed: ${CXX_PASSED}"
    echo "C++ tests failed: ${CXX_FAILED}"
    echo "C++ tests skipped: ${CXX_SKIPPED}"
    echo ""
    if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
        echo "Failed tests:"
        for item in "${FAILED_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Failed tests: none"
    fi
    echo ""
    if [[ ${#SKIPPED_TESTS[@]} -gt 0 ]]; then
        echo "Skipped tests:"
        for item in "${SKIPPED_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Skipped tests: none"
    fi
} >> "${SUMMARY_FILE}"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    cov_warn "One or more C++ tests failed; continuing to coverage generation."
fi

exit 0
