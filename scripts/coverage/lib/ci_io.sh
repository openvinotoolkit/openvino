#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -o pipefail

cov_init_io() {
    local workspace="${OV_WORKSPACE:-${GITHUB_WORKSPACE:-$(pwd)}}"
    COV_LOCAL_STATE_DIR="${COV_LOCAL_STATE_DIR:-${workspace}/.tmp/coverage-local}"
    COV_LOCAL_ENV_FILE="${COV_LOCAL_ENV_FILE:-${COV_LOCAL_STATE_DIR}/github_env}"
    COV_LOCAL_SUMMARY_FILE="${COV_LOCAL_SUMMARY_FILE:-${COV_LOCAL_STATE_DIR}/step_summary.md}"

    mkdir -p "${COV_LOCAL_STATE_DIR}"
    touch "${COV_LOCAL_ENV_FILE}" "${COV_LOCAL_SUMMARY_FILE}"

    export COV_LOCAL_STATE_DIR COV_LOCAL_ENV_FILE COV_LOCAL_SUMMARY_FILE
}

cov_env_output_file() {
    cov_init_io
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        printf '%s\n' "${GITHUB_ENV}"
    else
        printf '%s\n' "${COV_LOCAL_ENV_FILE}"
    fi
}

cov_summary_output_file() {
    cov_init_io
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        printf '%s\n' "${GITHUB_STEP_SUMMARY}"
    else
        printf '%s\n' "${COV_LOCAL_SUMMARY_FILE}"
    fi
}

cov_set_env() {
    local key="$1"
    local value="$2"
    local target

    target="$(cov_env_output_file)"
    printf '%s=%s\n' "${key}" "${value}" >> "${target}"
    export "${key}=${value}"
}

cov_append_summary() {
    local target

    target="$(cov_summary_output_file)"
    printf '%s\n' "$*" >> "${target}"
}

cov_load_local_env() {
    cov_init_io
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        return 0
    fi
    if [[ -s "${COV_LOCAL_ENV_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${COV_LOCAL_ENV_FILE}"
        set +a
    fi
}
