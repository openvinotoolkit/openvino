#!/usr/bin/env bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# exit when any command fails
set -e

usage() {
    echo "Build OpenVINO Runtime samples"
    echo
    echo "Options:"
    echo "  -h                       Print the help message"
    echo "  -b SAMPLES_BUILD_DIR     Specify the samples build directory"
    echo "  -i SAMPLES_INSTALL_DIR   Specify the samples install directory"
    echo
    exit 1
}

samples_type="$(basename "$(dirname "$(realpath "${BASH_SOURCE:-$0}")")")"
samples_build_dir="$HOME/openvino_${samples_type}_samples_build"
sample_install_dir=""

# parse command line options
while [[ $# -gt 0 ]]
do
case "$1" in
    -b | --build_dir)
    samples_build_dir="$2"
    shift
    ;;
    -i | --install_dir)
    sample_install_dir="$2"
    shift
    ;;
    -h | --help)
    usage
    ;;
    *)
    echo "Unrecognized option specified $1"
    usage
    ;;
esac
shift
done

error() {
    local code="${3:-1}"
    if [[ -n "$2" ]];then
        echo "Error on or near line $1: $2; exiting with status ${code}"
    else
        echo "Error on or near line $1; exiting with status ${code}"
    fi
    exit "${code}"
}
trap 'error ${LINENO}' ERR

SAMPLES_SOURCE_DIR="$( cd "$( dirname "$(realpath "${BASH_SOURCE:-$0}")" )" && pwd )"
printf "\nSetting environment variables for building samples...\n"

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    if [[ "$SAMPLES_SOURCE_DIR" = "/usr/share/openvino"* ]]; then
        true
    elif [ -e "$SAMPLES_SOURCE_DIR/../../setupvars.sh" ]; then
        setupvars_path="$SAMPLES_SOURCE_DIR/../../setupvars.sh"
        # shellcheck source=/dev/null
        source "$setupvars_path" || true
    else
        printf "Failed to set the environment variables automatically. To fix, run the following command:"
        printf "source <INTEL_OPENVINO_DIR>/setupvars.sh"
        printf "where INTEL_OPENVINO_DIR is the OpenVINO installation directory"
        exit 1
    fi
else
    # case for run with `sudo -E`
    # shellcheck source=/dev/null
    source "$INTEL_OPENVINO_DIR/setupvars.sh" || true
fi

# CentOS 7 has two packages: cmake of version 2.8 and cmake3. install_openvino_dependencies.sh installs cmake3
if command -v cmake3 &>/dev/null; then
    CMAKE_EXEC=cmake3
elif command -v cmake &>/dev/null; then
    CMAKE_EXEC=cmake
else
    printf "\n\nCMAKE is not installed. It is required to build OpenVINO Runtime samples. Please install it. \n\n"
    exit 1
fi

OS_PATH=$(uname -m)
NUM_THREADS=2

if [ "$OS_PATH" == "x86_64" ]; then
    OS_PATH="intel64"
    NUM_THREADS=8
fi

if [ -e "$samples_build_dir/CMakeCache.txt" ]; then
    rm -rf "$samples_build_dir/CMakeCache.txt"
fi

mkdir -p "$samples_build_dir"
cd "$samples_build_dir"
$CMAKE_EXEC -DCMAKE_BUILD_TYPE=Release "$SAMPLES_SOURCE_DIR"
$CMAKE_EXEC --build "$samples_build_dir" --config Release -- -j $NUM_THREADS

if [ "$sample_install_dir" != "" ]; then
    $CMAKE_EXEC -DCMAKE_INSTALL_PREFIX="$sample_install_dir" -DCOMPONENT=samples_bin -P cmake_install.cmake
    printf "\nBuild completed, you can find binaries for all samples in the %s/samples_bin subfolder.\n\n" "$sample_install_dir"
else
    printf "\nBuild completed, you can find binaries for all samples in the $samples_build_dir/%s/Release subfolder.\n\n" "$OS_PATH"
fi
