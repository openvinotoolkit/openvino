#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -o pipefail

COVERAGE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${COVERAGE_LIB_DIR}/ci_io.sh"

cov_repo_root() {
    local script_root

    if git_root=$(git rev-parse --show-toplevel 2>/dev/null); then
        printf '%s\n' "${git_root}"
        return 0
    fi

    script_root="$(cd "${COVERAGE_LIB_DIR}/../../.." && pwd)"
    printf '%s\n' "${script_root}"
}

cov_log() {
    printf '[coverage] %s\n' "$*"
}

cov_warn() {
    printf '[coverage][warn] %s\n' "$*"
}

cov_error() {
    printf '[coverage][error] %s\n' "$*" >&2
}

cov_require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        cov_error "Required command is missing: ${cmd}"
        return 1
    fi
}

cov_apply_profile() {
    case "${TEST_PROFILE}" in
        cpu)
            RUN_GPU_TESTS="false"
            RUN_NPU_TESTS="false"
            GPU_FLAGS="-DENABLE_INTEL_GPU=OFF -DENABLE_ONEDNN_FOR_GPU=OFF"
            NPU_FLAGS="-DENABLE_INTEL_NPU=OFF"
            ;;
        cpu_gpu)
            RUN_GPU_TESTS="true"
            RUN_NPU_TESTS="false"
            GPU_FLAGS="-DENABLE_INTEL_GPU=ON -DENABLE_ONEDNN_FOR_GPU=ON"
            NPU_FLAGS="-DENABLE_INTEL_NPU=OFF"
            ;;
        cpu_npu)
            RUN_GPU_TESTS="false"
            RUN_NPU_TESTS="true"
            GPU_FLAGS="-DENABLE_INTEL_GPU=OFF -DENABLE_ONEDNN_FOR_GPU=OFF"
            NPU_FLAGS="-DENABLE_INTEL_NPU=ON"
            ;;
        cpu_npu_gpu)
            RUN_GPU_TESTS="true"
            RUN_NPU_TESTS="true"
            GPU_FLAGS="-DENABLE_INTEL_GPU=ON -DENABLE_ONEDNN_FOR_GPU=ON"
            NPU_FLAGS="-DENABLE_INTEL_NPU=ON"
            ;;
        *)
            cov_error "Unsupported TEST_PROFILE: ${TEST_PROFILE}. Use one of: cpu, cpu_gpu, cpu_npu, cpu_npu_gpu"
            return 1
            ;;
    esac

    export RUN_GPU_TESTS RUN_NPU_TESTS GPU_FLAGS NPU_FLAGS
}

cov_init_defaults() {
    export OV_WORKSPACE="${OV_WORKSPACE:-${GITHUB_WORKSPACE:-$(cov_repo_root)}}"
    export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    export PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
    export PYTEST_XDIST_WORKERS="${PYTEST_XDIST_WORKERS:-1}"
    export JS_TEST_CONCURRENCY="${JS_TEST_CONCURRENCY:-1}"
    export TEST_PROFILE="${TEST_PROFILE:-cpu}"

    export BUILD_DIR="${BUILD_DIR:-${OV_WORKSPACE}/build}"
    export BUILD_JS_DIR="${BUILD_JS_DIR:-${OV_WORKSPACE}/build_js}"
    export INSTALL_PKG_DIR="${INSTALL_PKG_DIR:-${OV_WORKSPACE}/install_pkg}"
    export BIN_DIR="${BIN_DIR:-${OV_WORKSPACE}/bin/intel64/${CMAKE_BUILD_TYPE}}"
    export JS_DIR="${JS_DIR:-${OV_WORKSPACE}/src/bindings/js/node}"
    export MODEL_PATH="${MODEL_PATH:-${OV_WORKSPACE}/src/core/tests/models/ir/add_abc.xml}"

    cov_init_io
    cov_load_local_env
    cov_apply_profile
}

cov_print_profile() {
    cov_log "TEST_PROFILE=${TEST_PROFILE}"
    cov_log "RUN_GPU_TESTS=${RUN_GPU_TESTS}"
    cov_log "RUN_NPU_TESTS=${RUN_NPU_TESTS}"
    cov_log "GPU_FLAGS=${GPU_FLAGS}"
    cov_log "NPU_FLAGS=${NPU_FLAGS}"
}
