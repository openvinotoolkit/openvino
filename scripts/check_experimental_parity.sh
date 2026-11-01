#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Automation gate script to verify zero impact on upstream master when
# ENABLE_OV_EXPERIMENTAL=OFF, and (in --mode on) that an ON build legitimately
# exports the experimental BlockSparseAttention / DraftBlockMask symbols AND
# keeps the public include tree / experimental patch byte-aligned with upstream.
#
# Usage:
#   check_experimental_parity.sh [BUILD_DIR] [INSTALL_DIR]
#   check_experimental_parity.sh --mode [off|on] [BUILD_DIR] [INSTALL_DIR]
#   check_experimental_parity.sh --self-test
#
# Modes:
#   off (default): assert an OFF build cache, public include-tree purity, an empty
#                  install tree, zero experimental symbols exported, and the exact
#                  cm_sdpa_common.hpp.patch alignment.
#   on:            assert an ON build cache, the authoritative dynamic-symbol table
#                  exports BlockSparseAttention / DraftBlockMask from libopenvino.so,
#                  public include-tree purity, and the exact cm_sdpa_common.hpp.patch
#                  alignment.
#
# --self-test: verify the symbol-leak detection logic with fabricated input and the
#              cm_sdpa_common.hpp.patch matching logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="off"
BUILD_DIR="${REPO_ROOT}/build"
INSTALL_DIR="${BUILD_DIR}/install"
STRICT="${STRICT:-0}"

# Parse leading flags while preserving positional BUILD_DIR / INSTALL_DIR args.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-off}"
            shift 2
            ;;
        --self-test)
            SELF_TEST=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ "${MODE}" != "off" && "${MODE}" != "on" ]]; then
    echo "Invalid --mode '${MODE}'; expected 'off' or 'on'." >&2
    exit 2
fi

# Remaining positional args are BUILD_DIR and optionally INSTALL_DIR.
if [[ $# -ge 1 ]]; then
    BUILD_DIR="$1"
    INSTALL_DIR="${BUILD_DIR}/install"
    shift
fi
if [[ $# -ge 1 ]]; then
    INSTALL_DIR="$1"
    shift
fi

PASS=0
FAIL=0

log_pass() {
    echo -e "\033[32m[PASS]\033[0m $1"
    PASS=$((PASS + 1))
}

log_fail() {
    echo -e "\033[31m[FAIL]\033[0m $1"
    FAIL=$((FAIL + 1))
}

# -----------------------------------------------------------------------------
# Symbol filter. Reads a (possibly empty) leaked-symbol stream on stdin and prints
# only lines that match an experimental-path symbol. Extracted so --self-test can
# drive it with fabricated input.
# -----------------------------------------------------------------------------
gate_leaked_symbols() {
    grep -E "ov::op::experimental|ov::intel_gpu::experimental|BlockSparseAttention|DraftBlockMask" || true
}

# -----------------------------------------------------------------------------
# Gate 5: cm_sdpa_common.hpp exact experimental patch alignment. Byte-exactly
# applies the committed patch onto a pristine copy of the upstream file and
# asserts the result is identical to the in-tree fork.
# -----------------------------------------------------------------------------
check_cm_sdpa_patch() {
    local CM_SDPA_UPSTREAM="${REPO_ROOT}/src/plugins/intel_gpu/src/graph/impls/cm/include/cm_sdpa_common.hpp"
    local CM_SDPA_FORK="${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/graph/impls/cm/include/cm_sdpa_common.hpp"
    local CM_SDPA_PATCH="${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/cm_sdpa_common.hpp.patch"
    if [[ -f "${CM_SDPA_FORK}" && -f "${CM_SDPA_UPSTREAM}" && -f "${CM_SDPA_PATCH}" ]]; then
        local TMP_APPLIED
        TMP_APPLIED=$(mktemp)
        cp "${CM_SDPA_UPSTREAM}" "${TMP_APPLIED}"
        if ! patch --fuzz=0 -s --no-backup-if-mismatch -r - "${TMP_APPLIED}" < "${CM_SDPA_PATCH}" 2>/dev/null; then
            log_fail "experimental patch no longer applies to upstream cm_sdpa_common.hpp"
        elif cmp -s "${TMP_APPLIED}" "${CM_SDPA_FORK}"; then
            log_pass "cm_sdpa_common.hpp perfectly matches official experimental patch"
        else
            log_fail "cm_sdpa_common.hpp diverged from official experimental patch"
        fi
        rm -f "${TMP_APPLIED}" "${TMP_APPLIED}.rej" "${TMP_APPLIED}.orig"
    else
        if [[ "${STRICT}" == "1" ]]; then
            log_fail "Strict mode: cm_sdpa_common.hpp files or patch missing for Gate 5"
        else
            echo "  [INFO] cm_sdpa_common.hpp files or patch not found, skipping exact patch alignment check."
        fi
    fi
}
# -----------------------------------------------------------------------------
# --self-test: verify the symbol filter and the cm_sdpa_common.hpp patch logic.
# -----------------------------------------------------------------------------
run_self_test() {
    local SELF_PASS=0
    local SELF_FAIL=0

    step_pass() {
        echo -e "\033[32m[SELF-PASS]\033[0m $1"
        SELF_PASS=$((SELF_PASS + 1))
    }
    step_fail() {
        echo -e "\033[31m[SELF-FAIL]\033[0m $1"
        SELF_FAIL=$((SELF_FAIL + 1))
    }

    echo "=== [Parity Gate] Self-testing symbol-leak detection ==="

    # 1. A fabricated leaked experimental symbol MUST be flagged.
    local fake_input1="ov::op::experimental::v1::BlockSparseAttention::create()"
    local leaked1
    leaked1=$(printf '%s\n' "${fake_input1}" | gate_leaked_symbols)
    if [[ -n "${leaked1}" ]]; then
        step_pass "Symbol filter flags the injected leaked symbol '${leaked1}'"
    else
        step_fail "Symbol filter MISSED the injected leaked symbol '${fake_input1}'"
    fi

    # 2. A fabricated DraftBlockMask leaked symbol MUST be flagged.
    local fake_input2="ov::intel_gpu::experimental::DraftBlockMask::Impl()"
    local leaked2
    leaked2=$(printf '%s\n' "${fake_input2}" | gate_leaked_symbols)
    if [[ -n "${leaked2}" ]]; then
        step_pass "Symbol filter flags the injected leaked symbol '${leaked2}'"
    else
        step_fail "Symbol filter MISSED the injected leaked symbol '${fake_input2}'"
    fi

    # 3. A normal (non-experimental) symbol MUST be ignored.
    local clean_input="ov::op::v1::Softmax::get_axis()"
    local clean
    clean=$(printf '%s\n' "${clean_input}" | gate_leaked_symbols)
    if [[ -z "${clean}" ]]; then
        step_pass "Symbol filter correctly ignores the normal symbol '${clean_input}'"
    else
        step_fail "Symbol filter false-positived on the normal symbol '${clean_input}'"
    fi

    # 4. The cm_sdpa_common.hpp.patch matching logic: patch must apply cleanly
    #    and the result must be byte-identical to the in-tree fork.
    echo "=== [Parity Gate] Self-testing cm_sdpa_common.hpp patch logic ==="
    local CM_SDPA_UPSTREAM="${REPO_ROOT}/src/plugins/intel_gpu/src/graph/impls/cm/include/cm_sdpa_common.hpp"
    local CM_SDPA_FORK="${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/graph/impls/cm/include/cm_sdpa_common.hpp"
    local CM_SDPA_PATCH="${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/cm_sdpa_common.hpp.patch"
    if [[ -f "${CM_SDPA_FORK}" && -f "${CM_SDPA_UPSTREAM}" && -f "${CM_SDPA_PATCH}" ]]; then
        local TMP_APPLIED
        TMP_APPLIED=$(mktemp)
        cp "${CM_SDPA_UPSTREAM}" "${TMP_APPLIED}"
        if ! patch --fuzz=0 -s --no-backup-if-mismatch -r - "${TMP_APPLIED}" < "${CM_SDPA_PATCH}" 2>/dev/null; then
            step_fail "Self-test: experimental patch no longer applies to upstream cm_sdpa_common.hpp"
        elif cmp -s "${TMP_APPLIED}" "${CM_SDPA_FORK}"; then
            step_pass "cm_sdpa_common.hpp patch applies cleanly and matches the in-tree fork"
        else
            step_fail "Self-test: cm_sdpa_common.hpp diverged from official experimental patch"
        fi
        rm -f "${TMP_APPLIED}" "${TMP_APPLIED}.rej" "${TMP_APPLIED}.orig"
    else
        step_fail "Self-test: cm_sdpa_common.hpp files or patch missing"
    fi

    # 5. The pre-codegen CM include verifier's own --self-test must stay healthy.
    echo "=== [Parity Gate] Self-testing CM include verifier ==="
    if python3 "${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/cm_codegen/verify_cm_includes.py" --self-test > /dev/null 2>&1; then
        step_pass "verify_cm_includes.py --self-test passes"
    else
        step_fail "verify_cm_includes.py --self-test failed"
    fi

    echo "Self-test summary: ${SELF_PASS} Passed, ${SELF_FAIL} Failed."
    if [[ ${SELF_FAIL} -eq 0 ]]; then
        echo -e "\033[32m[SELF-TEST SUCCESS] Symbol filter + patch logic verified\033[0m"
        exit 0
    else
        echo -e "\033[31m[SELF-TEST ERROR] Symbol filter / patch logic is BROKEN\033[0m"
        exit 1
    fi
}

if [[ "${SELF_TEST:-0}" == "1" ]]; then
    run_self_test
fi

echo "=== [Parity Gate] Verifying Experimental Path Parity (mode=${MODE}) ==="
echo "Repo Root:    ${REPO_ROOT}"
echo "Build Dir:    ${BUILD_DIR}"
echo "Install Dir:  ${INSTALL_DIR}"

# -----------------------------------------------------------------------------
# Gate 1: Source Header & Tree Isolation Gate (both on & off)
# Ensure no experimental subdirectories or new experimental op headers exist in
# the public include tree.
# -----------------------------------------------------------------------------
echo "--> Gate 1: Checking public include tree cleanliness..."
LEAKED_EXP_DIR=$(find "${REPO_ROOT}/src/core/include" -type d -name "experimental" 2>/dev/null || true)
LEAKED_HEADERS=$(find "${REPO_ROOT}/src/core/include" -type f \( -name "*block_sparse_attention*" -o -name "*draft_block_mask*" \) 2>/dev/null || true)
if [[ -z "${LEAKED_EXP_DIR}" && -z "${LEAKED_HEADERS}" ]]; then
    log_pass "Public include directory (src/core/include) contains 0 experimental directories or leaked op headers"
else
    log_fail "Found experimental leakage in public include tree: dir='${LEAKED_EXP_DIR}', headers='${LEAKED_HEADERS}'"
fi

# -----------------------------------------------------------------------------
# Gate 2: Hook Verification & Build Cache Check
# -----------------------------------------------------------------------------
echo "--> Gate 2: Checking default option and build cache..."
if grep -q "ENABLE_OV_EXPERIMENTAL.*OFF" "${REPO_ROOT}/cmake/features.cmake"; then
    log_pass "ENABLE_OV_EXPERIMENTAL default is OFF in cmake/features.cmake"
else
    log_fail "ENABLE_OV_EXPERIMENTAL default is not OFF in cmake/features.cmake"
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    if grep -q "^ENABLE_OV_EXPERIMENTAL:BOOL=OFF" "${BUILD_DIR}/CMakeCache.txt"; then
        if [[ "${MODE}" == "off" ]]; then
            log_pass "Build cache has ENABLE_OV_EXPERIMENTAL:BOOL=OFF"
        else
            log_fail "Build cache has ENABLE_OV_EXPERIMENTAL:BOOL=OFF (must be ON for --mode on)"
        fi
    elif grep -q "^ENABLE_OV_EXPERIMENTAL:BOOL=ON" "${BUILD_DIR}/CMakeCache.txt"; then
        if [[ "${MODE}" == "on" ]]; then
            log_pass "Build cache has ENABLE_OV_EXPERIMENTAL:BOOL=ON (expected for --mode on)"
        else
            log_fail "Build cache has ENABLE_OV_EXPERIMENTAL:BOOL=ON (must be OFF for parity verification)"
        fi
    else
        log_fail "Build cache does not contain ENABLE_OV_EXPERIMENTAL setting"
    fi
else
    if [[ "${STRICT}" == "1" ]]; then
        log_fail "Strict mode: CMakeCache.txt not found in ${BUILD_DIR}"
    else
        echo "  [INFO] Build dir ${BUILD_DIR} CMakeCache.txt not found, skipping build cache check."
    fi
fi

# -----------------------------------------------------------------------------
# Gate 3: Installed Assets Check (mode off only)
# -----------------------------------------------------------------------------
if [[ "${MODE}" == "off" ]]; then
    echo "--> Gate 3: Checking install tree cleanliness (mode off)..."
    if [[ -d "${INSTALL_DIR}" ]]; then
        LEAKED_INSTALL=$(find "${INSTALL_DIR}" -type f \( -name "*block_sparse_attention*" -o -name "*draft_block_mask*" \) 2>/dev/null || true)
        LEAKED_EXP_INSTALL_DIR=$(find "${INSTALL_DIR}" -type d -name "experimental" 2>/dev/null || true)
        LEAKED_OPSET_FILE=$(find "${INSTALL_DIR}" -type f -name "experimental_opset1*" 2>/dev/null || true)
        if [[ -z "${LEAKED_INSTALL}" && -z "${LEAKED_EXP_INSTALL_DIR}" && -z "${LEAKED_OPSET_FILE}" ]]; then
            log_pass "Install tree contains 0 experimental op assets"
        else
            log_fail "Found experimental files in install tree: dirs='${LEAKED_EXP_INSTALL_DIR}', files='${LEAKED_INSTALL}', opset='${LEAKED_OPSET_FILE}'"
        fi
    else
        if [[ "${STRICT}" == "1" ]]; then
            log_fail "Strict mode: Install dir ${INSTALL_DIR} does not exist"
        else
            echo "  [INFO] Install dir ${INSTALL_DIR} does not exist, skipping physical install tree check."
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Gate 4: Exported Dynamic Symbols Check
# -----------------------------------------------------------------------------
echo "--> Gate 4: Checking exported dynamic symbols in libraries..."
CHECKED_LIBS=0

# Library set differs by mode.
if [[ "${MODE}" == "on" ]]; then
    LIBS_TO_CHECK=("libopenvino.so")
else
    LIBS_TO_CHECK=("libopenvino.so" "libopenvino_intel_gpu_plugin.so")
fi

for LIB_NAME in "${LIBS_TO_CHECK[@]}"; do
    # Locate the library, following symlinks to the real ELF file (the .so is a
    # symlink to .so.<version> in the build tree, so `-type f` + readlink resolution
    # is required to find a file nm -D can inspect).
    SO_FILE=""
    for ROOT in "${BUILD_DIR}" "${INSTALL_DIR}" "${REPO_ROOT}/bin" "${REPO_ROOT}/build"; do
        if [[ -d "${ROOT}" ]]; then
            FOUND=$(find "${ROOT}" -name "${LIB_NAME}*" -not -name "*.py" 2>/dev/null | while read -r f; do
                if [[ -e "${f}" ]]; then
                    REAL=$(readlink -f "${f}" 2>/dev/null || echo "${f}")
                    if [[ -f "${REAL}" ]]; then
                        echo "${REAL}"
                        break
                    fi
                fi
            done | head -n 1 || true)
            if [[ -n "${FOUND}" ]]; then
                SO_FILE="${FOUND}"
                break
            fi
        fi
    done
    if [[ -n "${SO_FILE}" && -f "${SO_FILE}" ]]; then
        CHECKED_LIBS=$((CHECKED_LIBS + 1))
        EXP_SYMBOLS=$(nm -D --defined-only "${SO_FILE}" 2>/dev/null | c++filt | gate_leaked_symbols || true)
        if [[ "${MODE}" == "on" ]]; then
            # ON mode: BlockSparseAttention / DraftBlockMask MUST be exported.
            BSA_PRESENT=$(nm -D --defined-only "${SO_FILE}" 2>/dev/null | c++filt | grep -E "BlockSparseAttention" || true)
            DBM_PRESENT=$(nm -D --defined-only "${SO_FILE}" 2>/dev/null | c++filt | grep -E "DraftBlockMask" || true)
            if [[ -n "${BSA_PRESENT}" && -n "${DBM_PRESENT}" ]]; then
                log_pass "${LIB_NAME} exports BlockSparseAttention and DraftBlockMask symbols"
            else
                log_fail "${LIB_NAME} missing expected experimental symbols: BSA='${BSA_PRESENT}', DBM='${DBM_PRESENT}'"
            fi
        else
            if [[ -z "${EXP_SYMBOLS}" ]]; then
                log_pass "${LIB_NAME} exports 0 experimental path symbols"
            else
                log_fail "${LIB_NAME} exports experimental symbols when OFF: ${EXP_SYMBOLS}"
            fi
        fi
    fi
done

if [[ ${CHECKED_LIBS} -eq 0 ]]; then
    if [[ "${STRICT}" == "1" ]]; then
        log_fail "Strict mode: Target libraries not found in ${BUILD_DIR}"
    else
        echo "  [INFO] Target libraries not found in ${BUILD_DIR}, skipping live symbol check."
    fi
fi

# -----------------------------------------------------------------------------
# Gate 5: CM SDPA exact experimental patch alignment (both on & off)
# -----------------------------------------------------------------------------
echo "--> Gate 5: Checking cm_sdpa_common.hpp exact experimental patch alignment..."
check_cm_sdpa_patch

# -----------------------------------------------------------------------------
# Gate 6: Pre-codegen CM include verifier health (both on & off)
# -----------------------------------------------------------------------------
echo "--> Gate 6: Checking CM codegen include verifier self-test..."
if python3 "${REPO_ROOT}/src/plugins/intel_gpu/src/experimental/cm_codegen/verify_cm_includes.py" --self-test > /dev/null 2>&1; then
    log_pass "verify_cm_includes.py --self-test passes"
else
    log_fail "verify_cm_includes.py --self-test failed (CM codegen include verification degraded)"
fi

echo "=========================================================="
echo "Parity Check Summary: ${PASS} Passed, ${FAIL} Failed."
if [[ ${FAIL} -eq 0 ]]; then
    echo -e "\033[32m[SUCCESS] Parity verification succeeded!\033[0m"
    exit 0
else
    echo -e "\033[31m[ERROR] Parity verification failed!\033[0m"
    exit 1
fi
