#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<'USAGE'
Usage: scripts/coverage/run_full.sh [options]

Options:
  --profile <cpu|cpu_gpu|cpu_npu|cpu_npu_gpu>  Hardware profile (default: cpu)
  --install-deps                               Run dependency installation phase
  --from <phase>                               Start from phase
  --to <phase>                                 End at phase
  --strict                                     Stop at first failed phase
  -h, --help                                   Show this help

Phases:
  install_deps
  configure
  build_install
  run_cpp_tests
  run_python_tests
  run_js_tests
  collect_cpp_coverage
  write_summary
  package_artifacts
USAGE
}

INSTALL_DEPS="false"
STRICT_MODE="false"
FROM_PHASE=""
TO_PHASE=""
TEST_PROFILE_ARG="cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            TEST_PROFILE_ARG="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS="true"
            shift
            ;;
        --from)
            FROM_PHASE="$2"
            shift 2
            ;;
        --to)
            TO_PHASE="$2"
            shift 2
            ;;
        --strict)
            STRICT_MODE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            cov_error "Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

export TEST_PROFILE="${TEST_PROFILE_ARG}"
cov_init_defaults

PHASES=(
    "install_deps"
    "configure"
    "build_install"
    "run_cpp_tests"
    "run_python_tests"
    "run_js_tests"
    "collect_cpp_coverage"
    "write_summary"
    "package_artifacts"
)

declare -A PHASE_SCRIPT=(
    [install_deps]="install_deps.sh"
    [configure]="configure_ov.sh"
    [build_install]="build_install.sh"
    [run_cpp_tests]="run_cpp_tests.sh"
    [run_python_tests]="run_python_tests.sh"
    [run_js_tests]="run_js_tests.sh"
    [collect_cpp_coverage]="collect_cpp_coverage.sh"
    [write_summary]="write_summary.sh"
    [package_artifacts]="package_artifacts.sh"
)

phase_index() {
    local needle="$1"
    local idx
    for idx in "${!PHASES[@]}"; do
        if [[ "${PHASES[idx]}" == "${needle}" ]]; then
            printf '%s\n' "${idx}"
            return 0
        fi
    done
    return 1
}

START_IDX=0
END_IDX=$((${#PHASES[@]} - 1))

if [[ -n "${FROM_PHASE}" ]]; then
    if ! START_IDX=$(phase_index "${FROM_PHASE}"); then
        cov_error "Unknown --from phase: ${FROM_PHASE}"
        exit 2
    fi
fi

if [[ -n "${TO_PHASE}" ]]; then
    if ! END_IDX=$(phase_index "${TO_PHASE}"); then
        cov_error "Unknown --to phase: ${TO_PHASE}"
        exit 2
    fi
fi

if [[ ${START_IDX} -gt ${END_IDX} ]]; then
    cov_error "Invalid range: --from phase is after --to phase"
    exit 2
fi

FAILED_PHASES=()

run_phase() {
    local phase="$1"
    local script_path="${SCRIPT_DIR}/${PHASE_SCRIPT[${phase}]}"

    if [[ "${phase}" == "install_deps" && "${INSTALL_DEPS}" != "true" ]]; then
        cov_log "Skipping install_deps phase (use --install-deps to enable)"
        return 0
    fi

    cov_log "Starting phase: ${phase}"
    if ! bash "${script_path}"; then
        cov_error "Phase failed: ${phase}"
        FAILED_PHASES+=("${phase}")
        if [[ "${STRICT_MODE}" == "true" ]]; then
            exit 1
        fi
    fi
}

for ((idx = START_IDX; idx <= END_IDX; idx++)); do
    run_phase "${PHASES[idx]}"
done

cov_log "Local coverage outputs:"
cov_log "  ${OV_WORKSPACE}/coverage.info"
cov_log "  ${OV_WORKSPACE}/python-coverage.xml"
cov_log "  ${OV_WORKSPACE}/js-lcov.info"
cov_log "  ${OV_WORKSPACE}/coverage-report/index.html"
cov_log "  ${COV_LOCAL_SUMMARY_FILE}"

if [[ ${#FAILED_PHASES[@]} -gt 0 ]]; then
    cov_error "Completed with failed phases: ${FAILED_PHASES[*]}"
    exit 1
fi

cov_log "Completed successfully"
