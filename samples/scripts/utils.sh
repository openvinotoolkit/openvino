#!/usr/bin/env bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]]; then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}

print_and_run() {
    printf 'Run'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
}
