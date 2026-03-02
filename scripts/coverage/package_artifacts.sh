#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

SUMMARY_FILE="$(cov_summary_output_file)"

if [[ -d "${OV_WORKSPACE}/coverage-report" && -f "${OV_WORKSPACE}/coverage.info" ]]; then
    (
        cd "${OV_WORKSPACE}"
        tar -czf coverage-report.tgz coverage-report coverage.info python-coverage.xml js-lcov.info || true
    )
    {
        echo ""
        echo "Artifact: \`coverage-report.tgz\` (download from Actions -> Artifacts, then extract and open \`coverage-report/index.html\`)."
    } >> "${SUMMARY_FILE}"
else
    {
        echo ""
        echo "coverage-report/ or coverage.info missing -> skipping artifact packaging."
    } >> "${SUMMARY_FILE}"
fi
