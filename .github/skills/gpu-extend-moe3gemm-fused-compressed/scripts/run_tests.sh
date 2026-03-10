#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Run GPU plugin test binaries for MoE development.
# All arguments are forwarded to the test binary.
#
# Usage:  ./run_tests.sh <binary> [gtest_args...]
#   e.g.: ./run_tests.sh ov_gpu_unit_tests
#         ./run_tests.sh ov_gpu_unit_tests --gtest_filter='*moe_3gemm*'
#         ./run_tests.sh ov_gpu_func_tests --gtest_filter='*MoE3GemmCompressed*'
#
# Set BUILD_DIR to override the build directory (default: $SRC_DIR/build/Release).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build/Release}"

BINARY="${1:?Usage: $0 <binary> [gtest_args...]}"
shift

"$BUILD_DIR/bin/$BINARY" "$@"
