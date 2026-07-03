#!/bin/bash

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Install CMake as a build dependency.

set -o errexit
set -o errtrace
set -o pipefail
set -o nounset

cleanup() {
    rm --force cmake.sh
}

trap cleanup EXIT

if [[ -z "${1:-}" ]]; then
    echo "Error: CMAKE_VERSION argument was not provided." >&2
    echo "Usage: $(basename "$0") <cmake_version>" >&2
    exit 1
fi

CMAKE_VERSION="${1}"
ARCHITECTURE=$(uname --machine)

wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCHITECTURE}.sh" --output-document=cmake.sh

sh cmake.sh --skip-license --prefix=/usr/local
