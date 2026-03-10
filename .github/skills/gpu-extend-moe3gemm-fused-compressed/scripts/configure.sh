#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Configure CMake build for GPU plugin MoE development.
# Run once from the OpenVINO source root, or set SRC_DIR / BUILD_DIR as needed.
#
# Usage:  SRC_DIR=/path/to/openvino BUILD_DIR=/path/to/build ./configure.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at .github/skills/<skill>/scripts/ → source root is 4 levels up
SRC_DIR="${SRC_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build/Release}"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
    -DENABLE_INTEL_GPU=TRUE \
    -DENABLE_INTEL_NPU=FALSE \
    -DENABLE_TESTS=TRUE \
    -DENABLE_PYTHON=TRUE \
    -DENABLE_DEBUG_CAPS=TRUE \
    -DENABLE_GPU_DEBUG_CAPS=TRUE \
    -DENABLE_CPPLINT=FALSE \
    -DENABLE_CLANG_FORMAT=FALSE \
    -DENABLE_OPENVINO_DEBUG=FALSE \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    "-DCMAKE_CXX_FLAGS=-Wno-deprecated -Wno-deprecated-declarations" \
    "-DCMAKE_C_FLAGS=-Wno-deprecated -Wno-deprecated-declarations" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
    --no-warn-unused-cli \
    -S "$SRC_DIR" \
    -B "$BUILD_DIR" \
    -G Ninja
