#!/bin/bash

# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

NGRAPH_ONNX_REPO="https://github.com/NervanaSystems/ngraph-onnx"
CI_PATH="$(pwd)"
CI_ROOT=".ci/onnx/jenkins"
REPO_ROOT="${CI_PATH%$CI_ROOT}"
DOCKER_CONTAINER="ngraph-onnx_ci_reproduction"

# Function run() builds image with requirements needed to build ngraph and run onnx tests, runs container and executes tox tests
function run() {
    set -x
    set -e

    cd ./dockerfiles
    docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f=./ubuntu_16_04.dockerfile -t ngraph-onnx:ubuntu-16_04 .

    cd "${CI_PATH}"
    if [[ -z $(docker ps -a | grep -i "${DOCKER_CONTAINER}") ]];
    then
        docker run -h "$(hostname)" --privileged --name "${DOCKER_CONTAINER}" -v "${REPO_ROOT}":/root \
            -d ngraph-onnx:ubuntu-16_04 tail -f /dev/null
        BUILD="TRUE"
    fi

    if [[ "${BUILD}" == "TRUE" ]];
    then
        BUILD_NGRAPH_CMD='cd /root && \
            mkdir -p ./build && \
            cd ./build && \
            cmake ../ -DNGRAPH_TOOLS_ENABLE=FALSE -DNGRAPH_UNIT_TEST_ENABLE=FALSE \
            -DNGRAPH_ONNX_IMPORT_ENABLE=TRUE -DCMAKE_INSTALL_PREFIX=/root/ngraph_dist && \
            make -j $(lscpu --parse=CORE | grep -v '"'#'"' | sort | uniq | wc -l) && \
            make install && \
            cd /root/python && \
            if [[ -z $(ls /root/ngraph-onnx 2>/dev/null) ]]; then
                git clone --recursive https://github.com/pybind/pybind11.git;
            fi
            export PYBIND_HEADERS_PATH=/root/python/pybind11 && \
            export NGRAPH_CPP_BUILD_PATH=/root/ngraph_dist && \
            export NGRAPH_ONNX_IMPORT_ENABLE=TRUE && \
            python3 setup.py bdist_wheel && \
            cd /root'
        docker exec "${DOCKER_CONTAINER}" bash -c "${BUILD_NGRAPH_CMD}"
    fi

    CLONE_CMD='cd /root &&\
        if [[ -z $(ls /root/ngraph-onnx 2>/dev/null) ]]; then
            git clone '"${NGRAPH_ONNX_REPO}"';
        fi'
    docker exec "${DOCKER_CONTAINER}" bash -c "${CLONE_CMD}"
    NGRAPH_WHL=$(docker exec ${DOCKER_CONTAINER} find /root/python/dist/ -name "ngraph*.whl")
    docker exec -e TOX_INSTALL_NGRAPH_FROM="${NGRAPH_WHL}" -e NGRAPH_BACKEND=CPU "${DOCKER_CONTAINER}" tox -c /root/ngraph-onnx/
    docker exec -e TOX_INSTALL_NGRAPH_FROM="${NGRAPH_WHL}" -e NGRAPH_BACKEND=INTEPRETER "${DOCKER_CONTAINER}" tox -c /root/ngraph-onnx/
}

# Function cleanup() removes items related to nGraph, created during script execution
function cleanup_ngraph() {
    set -x

    docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf /root/build/* /root/ngraph_dist /root/python/dist'
}

# Function cleanup() removes items created during script execution
function cleanup() {
    set -x

    docker exec "${DOCKER_CONTAINER}" bash -c "rm -rf /root/ngraph_dist /root/ngraph-onnx/.tox /root/ngraph-onnx/.onnx \
                    /root/ngraph-onnx/__pycache__ /root/ngraph-onnx/ngraph_onnx.egg-info /root/ngraph-onnx/cpu_codegen"
    docker exec "${DOCKER_CONTAINER}" bash -c 'rm -rf $(find /root/ -user root)'
    docker rm -f "${DOCKER_CONTAINER}"
}

PATTERN='[-a-zA-Z0-9_]*='
for i in "$@"
do
    case $i in
        --help*)
            printf "Script builds nGraph and runs tox tests inside docker container.
            Every execution after first run is going to run tox tests again.
            To rebuild nGraph and run tests again use --rebuild parameter.

            Following parameters are available:

            --help      displays this message
            --cleanup   removes docker container and files created during script execution
            --rebuild   rebuilds nGraph and runs tox tests
            "
            exit 0
        ;;
        --cleanup*)
            cleanup
            exit 0
        ;;
        --rebuild*)
            cleanup_ngraph
            BUILD="TRUE"
        ;;
    esac
done

set -x

run
