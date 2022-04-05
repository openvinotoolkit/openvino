#!/usr/bin/env bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

usage() {
    echo "Build OpenVINO Runtime samples"
    echo
    echo "Options:"
    echo "  -h                       Print the help message"
    echo "  -b SAMPLE_BUILD_DIR      Specify the sample build directory"
    echo "  -i SAMPLE_INSTALL_DIR    Specify the sample install directory"
    echo
    exit 1
}

samples_type=$(basename "$( dirname "${BASH_SOURCE[0]-$0}" )" )
build_dir="$HOME/inference_engine_${samples_type}_samples_build"
sample_install_dir=""

# parse command line options
while [[ $# -gt 0 ]]
do
case "$1" in
    -b | --build_dir)
    build_dir="$2"
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

SAMPLES_PATH="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" && pwd )"

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build OpenVINO Runtime samples. Please install it. \n\n"
    exit 1
fi

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ "$OS_PATH" == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS="-j8"
fi

if [ -e "$build_dir/CMakeCache.txt" ]; then
  rm -rf "$build_dir/CMakeCache.txt"
fi

mkdir -p "$build_dir"
cd "$build_dir"
cmake -DCMAKE_BUILD_TYPE=Release "$SAMPLES_PATH"
make $NUM_THREADS

if [ "$sample_install_dir" != "" ]; then
    cmake -DCMAKE_INSTALL_PREFIX="$sample_install_dir" -DCOMPONENT=samples_bin -P cmake_install.cmake
    printf "\nBuild completed, you can find binaries for all samples in the %s/samples_bin subfolder.\n\n" "$sample_install_dir"
else
    printf "\nBuild completed, you can find binaries for all samples in the $build_dir/%s/Release subfolder.\n\n" "$OS_PATH"
fi

