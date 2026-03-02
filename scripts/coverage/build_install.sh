#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

cov_log "Building OpenVINO"
cmake --build "${BUILD_DIR}" --parallel "${PARALLEL_JOBS}" --config "${CMAKE_BUILD_TYPE}"

cov_log "Installing OpenVINO Python wheels"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PKG_DIR}" --component python_wheels --config "${CMAKE_BUILD_TYPE}"

cov_log "Installing OpenVINO runtime and tests"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PKG_DIR}" --config "${CMAKE_BUILD_TYPE}"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PKG_DIR}" --component tests --config "${CMAKE_BUILD_TYPE}"

if [[ "${RUN_NPU_TESTS}" == "true" ]]; then
    JS_NPU_FLAG="-DENABLE_INTEL_NPU=ON"
else
    JS_NPU_FLAG="-DENABLE_INTEL_NPU=OFF"
fi

cov_log "Building and installing JS addon runtime"
cmake -S "${OV_WORKSPACE}" -B "${BUILD_JS_DIR}" \
    -GNinja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCPACK_GENERATOR=NPM \
    -DENABLE_SYSTEM_TBB=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_WHEEL=OFF \
    -DENABLE_PYTHON=OFF \
    -DENABLE_INTEL_GPU=OFF \
    ${JS_NPU_FLAG} \
    -DENABLE_JS=ON \
    -DENABLE_COVERAGE=ON \
    -DCMAKE_INSTALL_PREFIX="${JS_DIR}/bin"

cmake --build "${BUILD_JS_DIR}" --parallel "${PARALLEL_JOBS}" --config "${CMAKE_BUILD_TYPE}"
cmake --install "${BUILD_JS_DIR}" --prefix "${JS_DIR}/bin" --config "${CMAKE_BUILD_TYPE}"

cov_log "Installing built OpenVINO wheel"
WHEEL_PATH=$(find "${INSTALL_PKG_DIR}/wheels" -maxdepth 1 -type f -name 'openvino-*.whl' | head -n 1 || true)
if [[ -z "${WHEEL_PATH}" ]]; then
    cov_error "OpenVINO wheel not found in ${INSTALL_PKG_DIR}/wheels"
    exit 1
fi
python3 -m pip install --force-reinstall "${WHEEL_PATH}"

cov_log "Binaries in ${BIN_DIR}"
ls -la "${BIN_DIR}"
