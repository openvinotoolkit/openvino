#!/usr/bin/env bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

cov_init_defaults

SUDO=""
if [[ ${EUID} -ne 0 ]]; then
    SUDO="sudo"
fi

cov_log "Installing system dependencies"
${SUDO} apt --assume-yes update
${SUDO} -E "${OV_WORKSPACE}/install_build_dependencies.sh"
${SUDO} apt --assume-yes install lcov wget pigz xvfb clang-14 libclang-14-dev clinfo ca-certificates

cov_log "Installing Python dependencies"
python3 -m pip install --upgrade pip
python3 -m pip install pyyaml pytest pytest-cov pytest-xdist[psutil]
python3 -m pip install -r "${OV_WORKSPACE}/src/bindings/python/wheel/requirements-dev.txt"
python3 -m pip install -r "${OV_WORKSPACE}/src/frontends/paddle/tests/requirements.txt"
python3 -m pip install -r "${OV_WORKSPACE}/src/frontends/onnx/tests/requirements.txt"
python3 -m pip install -r "${OV_WORKSPACE}/src/frontends/tensorflow/tests/requirements.txt"
python3 -m pip install -r "${OV_WORKSPACE}/src/frontends/tensorflow_lite/tests/requirements.txt"
python3 -m pip install -r "${OV_WORKSPACE}/tests/requirements_jax"
