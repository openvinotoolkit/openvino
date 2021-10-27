#!/usr/bin/env bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

usage() {
    echo "Build inference engine samples"
    echo
    echo "Options:"
    echo "  -h                      Print the help message"
    echo "  -o OUTPUT_DIR           Specify the output directory"
    echo
    exit 1
}

samples_type=$(basename "$PWD")
build_dir="$HOME/inference_engine_${samples_type}_samples_build"

# parse command line options
while [[ $# -gt 0 ]]
do
case "$1" in
    -h | -help | --help)
    usage
    ;;
    -o | -output_dir | --output_dir)
    build_dir="$2"
    shift
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

printf "\nSetting environment variables for building samples...\n"

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    if [ -e "$SAMPLES_PATH/../../setupvars.sh" ]; then
        setvars_path="$SAMPLES_PATH/../../setupvars.sh"
    else
        printf "Error: Failed to set the environment variables automatically. To fix, run the following command:\n source <INSTALL_DIR>/setupvars.sh\n where INSTALL_DIR is the OpenVINO installation directory.\n\n"
        exit 1
    fi
    if ! source "$setvars_path" ; then
        printf "Unable to run ./setupvars.sh. Please check its presence. \n\n"
        exit 1
    fi
else
    # case for run with `sudo -E` 
    source "$INTEL_OPENVINO_DIR/setupvars.sh"
fi

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. \n\n"
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

printf "\nBuild completed, you can find binaries for all samples in the $build_dir/%s/Release subfolder.\n\n" "$OS_PATH"
