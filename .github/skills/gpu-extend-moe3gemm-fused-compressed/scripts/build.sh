#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build a CMake target for GPU plugin MoE development.
#
# Usage:  ./build.sh <target>
#   e.g.: ./build.sh openvino_intel_gpu_plugin
#         ./build.sh ov_gpu_unit_tests
#         ./build.sh ov_gpu_func_tests
#
# Set BUILD_DIR to override the build directory (default: $SRC_DIR/build/Release).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build/Release}"

cmake --build "$BUILD_DIR" --config Release --target "${1:?Usage: $0 <target>}" -j 24
