#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

SRC_DIR="${OV_WORKSPACE}"
REPORT_DIR="${OV_WORKSPACE}/coverage-report"
MAIN_CPP_INFO="${OV_WORKSPACE}/coverage-cpp-main.info"
JS_CPP_INFO="${OV_WORKSPACE}/coverage-cpp-js.info"

cov_log "Capturing coverage from ${BUILD_DIR} (base=${SRC_DIR})"
lcov --capture \
    --directory "${BUILD_DIR}" \
    --build-directory "${BUILD_DIR}" \
    --base-directory "${SRC_DIR}" \
    --output-file "${MAIN_CPP_INFO}" \
    --no-external \
    --rc geninfo_unexecuted_blocks=1 \
    --ignore-errors mismatch,negative,unused,gcov

if find "${BUILD_JS_DIR}" -name '*.gcda' -print -quit | grep -q .; then
    cov_log "Capturing C/C++ coverage from ${BUILD_JS_DIR}"
    lcov --capture \
        --directory "${BUILD_JS_DIR}" \
        --build-directory "${BUILD_JS_DIR}" \
        --base-directory "${SRC_DIR}" \
        --output-file "${JS_CPP_INFO}" \
        --no-external \
        --rc geninfo_unexecuted_blocks=1 \
        --ignore-errors mismatch,negative,unused,gcov
    lcov -a "${MAIN_CPP_INFO}" -a "${JS_CPP_INFO}" -o "${OV_WORKSPACE}/coverage.info"
else
    cov_warn "No .gcda files found in ${BUILD_JS_DIR}, skipping JS native C++ capture"
    cp "${MAIN_CPP_INFO}" "${OV_WORKSPACE}/coverage.info"
fi

cov_log "Applying exclude patterns (tests, thirdparty, protobuf generated)"
lcov --remove "${OV_WORKSPACE}/coverage.info" \
    --ignore-errors unused,mismatch \
    "${SRC_DIR}/build/*" \
    "${SRC_DIR}/build_js/*" \
    "${SRC_DIR}/*.pb.cc" \
    "${SRC_DIR}/*.pb.h" \
    "${SRC_DIR}/*/tests/*" \
    "${SRC_DIR}/tests/*" \
    "${SRC_DIR}/docs/*" \
    "${SRC_DIR}/samples/*" \
    "${SRC_DIR}/tools/*" \
    "${SRC_DIR}/src/bindings/js/node/tests/*" \
    "${SRC_DIR}/src/bindings/python/tests/*" \
    "${SRC_DIR}/thirdparty/*" \
    -o "${OV_WORKSPACE}/coverage.info"

rm -f "${MAIN_CPP_INFO}" "${JS_CPP_INFO}" || true

cov_log "Checking for remaining build-path entries in coverage.info"
grep -m 5 "^SF:${SRC_DIR}/build/" "${OV_WORKSPACE}/coverage.info" || true
grep -m 5 "^SF:${SRC_DIR}/build_js/" "${OV_WORKSPACE}/coverage.info" || true

mkdir -p "${REPORT_DIR}"
genhtml "${OV_WORKSPACE}/coverage.info" \
    --output-directory "${REPORT_DIR}" \
    --prefix "${SRC_DIR}" \
    --synthesize-missing
