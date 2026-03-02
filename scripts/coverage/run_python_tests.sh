#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set +e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

CONFIG_FILE="${SCRIPT_DIR}/config/tests_python.yml"
CONFIG_READER="${SCRIPT_DIR}/lib/coverage_config.py"
TESTS_DIR="${INSTALL_PKG_DIR}/tests"
SRC_PY_TESTS_DIR="${OV_WORKSPACE}/src/bindings/python/tests"
ONNX_PY_TESTS_DIR="${OV_WORKSPACE}/src/frontends/onnx/tests/tests_python"
WORKSPACE_LAYER_TESTS_DIR="${OV_WORKSPACE}/tests/layer_tests"
PY_COV_CONFIG="${OV_WORKSPACE}/.python_coverage_ci.rc"
SUMMARY_FILE="$(cov_summary_output_file)"
PY_TOTAL_EXECUTED=0
FAILED_PY_TESTS=()
SKIPPED_PY_TESTS=()

if [[ ! -f "${CONFIG_FILE}" ]]; then
    cov_error "Python tests config file is missing: ${CONFIG_FILE}"
    exit 1
fi

python3 -m pip install -r "${TESTS_DIR}/bindings/python/requirements_test.txt"
python3 -m pip install -r "${TESTS_DIR}/layer_tests/requirements.txt"
python3 -m pip install -r "${TESTS_DIR}/requirements_onnx"
python3 -m pip install -r "${TESTS_DIR}/requirements_jax"

export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TESTS_DIR}/python:${PYTHONPATH:-}"
export TESTS_DIR SRC_PY_TESTS_DIR ONNX_PY_TESTS_DIR WORKSPACE_LAYER_TESTS_DIR PY_COV_CONFIG

cat > "${PY_COV_CONFIG}" << 'PYCOV'
[run]
omit =
    */tests/*
    */thirdparty/*
    */docs/*
    */samples/*
    */tools/*
    */src/bindings/js/node/tests/*
    */src/bindings/python/tests/*
    *.pb.cc
    *.pb.h
PYCOV

coverage erase

expand_vars() {
    local input="$1"
    python3 - "$input" <<'PY'
import os
import sys
print(os.path.expandvars(sys.argv[1]))
PY
}

run_shell_test() {
    local test_name="$1"
    local cmd="$2"

    PY_TOTAL_EXECUTED=$((PY_TOTAL_EXECUTED + 1))
    echo "========== Running Python test: ${test_name} =========="
    bash -lc "${cmd}"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        FAILED_PY_TESTS+=("${test_name} (exit ${rc})")
    fi
}

while IFS= read -r raw_line; do
    IFS='|' read -r test_name enabled skip_reason kind target args test_env command <<< "${raw_line//$'\t'/|}"
    [[ -z "${test_name}" ]] && continue

    if [[ "${enabled}" != "1" ]]; then
        SKIPPED_PY_TESTS+=("${test_name} (${skip_reason})")
        continue
    fi

    expanded_target="$(expand_vars "${target}")"
    expanded_args="$(expand_vars "${args}")"
    expanded_env="$(expand_vars "${test_env}")"
    expanded_command="$(expand_vars "${command}")"

    case "${kind}" in
        pytest)
            target_q=$(printf '%q' "${expanded_target}")
            full_cmd="python3 -m pytest -ra --durations=50 ${target_q}"
            if [[ -n "${expanded_args}" ]]; then
                full_cmd+=" ${expanded_args}"
            fi
            if [[ -n "${expanded_env}" ]]; then
                full_cmd="${expanded_env} ${full_cmd}"
            fi
            run_shell_test "${test_name}" "${full_cmd}"
            ;;
        pytest_if_dir)
            if [[ -d "${expanded_target}" ]]; then
                target_q=$(printf '%q' "${expanded_target}")
                full_cmd="python3 -m pytest -ra --durations=50 ${target_q}"
                if [[ -n "${expanded_args}" ]]; then
                    full_cmd+=" ${expanded_args}"
                fi
                if [[ -n "${expanded_env}" ]]; then
                    full_cmd="${expanded_env} ${full_cmd}"
                fi
                run_shell_test "${test_name}" "${full_cmd}"
            else
                cov_warn "Skipping Python test group '${test_name}' (missing: ${expanded_target})"
                SKIPPED_PY_TESTS+=("${test_name} (missing path)")
            fi
            ;;
        command)
            if [[ -n "${expanded_env}" ]]; then
                expanded_command="${expanded_env} ${expanded_command}"
            fi
            run_shell_test "${test_name}" "${expanded_command}"
            ;;
        *)
            cov_warn "Unknown Python test kind '${kind}' for '${test_name}', skipping"
            SKIPPED_PY_TESTS+=("${test_name} (unknown kind: ${kind})")
            ;;
    esac
done < <(python3 "${CONFIG_READER}" --suite python --profile "${TEST_PROFILE}" --config "${CONFIG_FILE}")

coverage xml -o "${OV_WORKSPACE}/python-coverage.xml"

PY_FAILED=${#FAILED_PY_TESTS[@]}
PY_SKIPPED=${#SKIPPED_PY_TESTS[@]}
PY_PASSED=$((PY_TOTAL_EXECUTED - PY_FAILED))
cov_set_env "PY_TESTS_TOTAL" "$((PY_TOTAL_EXECUTED + PY_SKIPPED))"
cov_set_env "PY_TESTS_PASSED" "${PY_PASSED}"
cov_set_env "PY_TESTS_FAILED" "${PY_FAILED}"
cov_set_env "PY_TESTS_SKIPPED" "${PY_SKIPPED}"

{
    echo ""
    echo "## Python coverage test execution summary"
    echo "Python tests executed: ${PY_TOTAL_EXECUTED}"
    echo "Python tests passed: ${PY_PASSED}"
    echo "Python tests failed: ${PY_FAILED}"
    echo "Python tests skipped: ${PY_SKIPPED}"
    echo ""
    if [[ ${#FAILED_PY_TESTS[@]} -gt 0 ]]; then
        echo "Failed tests:"
        for item in "${FAILED_PY_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Failed tests: none"
    fi
    echo ""
    if [[ ${#SKIPPED_PY_TESTS[@]} -gt 0 ]]; then
        echo "Skipped tests:"
        for item in "${SKIPPED_PY_TESTS[@]}"; do
            echo "- ${item}"
        done
    else
        echo "Skipped tests: none"
    fi
} >> "${SUMMARY_FILE}"

if [[ ${#FAILED_PY_TESTS[@]} -gt 0 ]]; then
    cov_warn "One or more Python tests failed; continuing to coverage generation."
fi

exit 0
