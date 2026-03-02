#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults
cov_print_profile

read -r -a GPU_FLAGS_ARR <<< "${GPU_FLAGS}"
read -r -a NPU_FLAGS_ARR <<< "${NPU_FLAGS}"

cmake -S "${OV_WORKSPACE}" -B "${BUILD_DIR}" \
    -GNinja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_JS=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_FUNCTIONAL_TESTS=ON \
    -DENABLE_OV_ONNX_FRONTEND=ON \
    -DENABLE_OV_PADDLE_FRONTEND=ON \
    -DENABLE_OV_TF_FRONTEND=ON \
    -DENABLE_OV_TF_LITE_FRONTEND=ON \
    -DENABLE_STRICT_DEPENDENCIES=OFF \
    -DENABLE_COVERAGE=ON \
    -DCMAKE_C_COMPILER="${CC:-gcc}" \
    -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_LINKER_LAUNCHER=ccache \
    -DCMAKE_CXX_LINKER_LAUNCHER=ccache \
    -DENABLE_SYSTEM_SNAPPY=ON \
    "${GPU_FLAGS_ARR[@]}" \
    "${NPU_FLAGS_ARR[@]}"
