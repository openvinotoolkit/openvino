#!/bin/bash

# Copyright (c) 2018 Intel Corporation
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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

SAMPLES_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z "${InferenceEngine_DIR}" ]]; then
    printf "\nInferenceEngine_DIR environment variable is not set. Trying to find setupvars.sh to set it. \n"
    
    setvars_path=$SAMPLES_PATH/../..
    if [ -e "$setvars_path/inference_engine/bin/setvars.sh" ]; then # for Intel Deep Learning Deployment Toolkit package
        setvars_path="$setvars_path/inference_engine/bin/setvars.sh"
    elif [ -e "$setvars_path/../bin/setupvars.sh" ]; then # for OpenVINO package
        setvars_path="$setvars_path/../bin/setupvars.sh"
    elif [ -e "$setvars_path/../setupvars.sh" ]; then
        setvars_path="$setvars_path/../setupvars.sh"
    else
        printf "Error: setupvars.sh is not found in hardcoded paths. \n\n"
        exit 1
    fi 
    if ! source $setvars_path ; then
        printf "Unable to run ./setupvars.sh. Please check its presence. \n\n"
        exit 1
    fi
fi

if ! command -v cmake &>/dev/null; then
    printf "\n\nCMAKE is not installed. It is required to build Inference Engine samples. Please install it. \n\n"
    exit 1
fi

build_dir=$HOME/inference_engine_samples_build
mkdir -p $build_dir
cd $build_dir
cmake -DCMAKE_BUILD_TYPE=Release $SAMPLES_PATH
make -j8

printf "\nBuild completed, you can find binaries for all samples in the $HOME/inference_engine_samples_build/intel64/Release subfolder.\n\n"
