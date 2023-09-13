#!/bin/bash

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

abs_path () {
    script_path=$(eval echo "$1")
    directory=$(dirname "$script_path")
    echo "$(cd "$directory" || exit; pwd -P)";
}

SCRIPT_DIR="$(abs_path "${BASH_SOURCE[0]}")" >/dev/null 2>&1
INSTALLDIR="${SCRIPT_DIR}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${INSTALLDIR}/../ov_runtime/runtime/lib/intel64/
