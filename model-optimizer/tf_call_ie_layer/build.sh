# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

get_tf_version()
{
    tf_dir=$1
    TF_VERSION=$(cd $tf_dir 2>/dev/null; git describe --abbrev=0 --tags 2>/dev/null)
    if [ $? != 0 ]; then
        echo "ERROR: failed to determine TensorFlow version."
        echo "Please, set environment variable TF_VERSION with TensorFlow version."
        echo "For example, 'export TF_VERSION=1.5'"
        exit 1
    else
        TF_VERSION=$(echo $TF_VERSION | sed 's/v\([0-9]\+\.[0-9]\+\).*/\1/')
        echo "Automatically determined version of TensorFlow as: $TF_VERSION"
    fi
}

determine_layer_BUILD_file()
{
    if [ $TF_VERSION_MAJOR -gt 1 ]; then
        LAYER_BUILD_FILE=BUILD_1_4_higher
        echo "TensorFlow major version is higher than 1, so use BUILD configuraion file as for 1.4."
        echo "WARNING: this TensorFlow version has not been tested!"
    else
        if [ $TF_VERSION_MINOR -ge 4 ]; then
            echo "TensorFlow minor version is higher than 3, so use BUILD configuraion file as for 1.4"
            LAYER_BUILD_FILE=BUILD_1_4_higher
        else
            if [ $TF_VERSION_MINOR -ge 2 ]; then
                echo "TensorFlow minor version is between 2 and 3, so use BUILD configuraion file as for 1.2"
                LAYER_BUILD_FILE=BUILD_1_2_to_1_3
            else
                echo "ERROR: TensorFlow version is not supported. Versions 1.2-1.5 are supported"
                exit 1
            fi
        fi
    fi
}

THIS_DIR=`dirname "$0"`
if echo "$THIS_DIR" | grep -q -s ^/ || echo "$THIS_DIR" | grep -q -s ^~ ; then
   THIS_ABSOLUTE_DIR="$THIS_DIR"
else
   THIS_ABSOLUTE_DIR="`pwd`/$THIS_DIR"
fi

set -e # exit if something goes wrong
if [ "x$INTEL_OPENVINO_DIR" = "x" ]; then
    echo "ERROR: INTEL_OPENVINO_DIR environment variable is not set"
    echo "Please, run the 'source <OpenVINO_install_dir>/bin/setupvars.sh'"
    exit 1
fi

if [ "x$TF_ROOT_DIR" == 'x' ]; then
    echo "ERROR: TF_ROOT_DIR environment variable is not set"
    echo "Please, set TF_ROOT_DIR environment variable which points to directory with cloned TF"
    exit 1
fi

IE_HEADERS_SRC_DIR=$INTEL_OPENVINO_DIR/inference_engine/include
if [ ! -e $IE_HEADERS_SRC_DIR ]; then
    echo "ERROR: Inference Engine headers files '$IE_HEADERS_SRC_DIR' doesn't exist"
    exit 1
fi

IE_HEADERS_DEST_DIR=$TF_ROOT_DIR/third_party/inference_engine
if [ -e $IE_HEADERS_DEST_DIR ]; then
    echo "Removing old version of IE headers files from '$IE_HEADERS_DEST_DIR'"
    rm -rf $IE_HEADERS_DEST_DIR
fi

if [ "x$TF_VERSION" = "x" ]; then
    get_tf_version $TF_ROOT_DIR
fi

TF_VERSION_MINOR=$(echo $TF_VERSION | awk -F. '{print $2}')
TF_VERSION_MAJOR=$(echo $TF_VERSION | awk -F. '{print $1}')
echo "TensorFlow major version: $TF_VERSION_MAJOR"
echo "TensorFlow minor version: $TF_VERSION_MINOR"

determine_layer_BUILD_file

mkdir -p $IE_HEADERS_DEST_DIR
cp -r ${IE_HEADERS_SRC_DIR} ${IE_HEADERS_DEST_DIR}
cp $THIS_ABSOLUTE_DIR/inference_engine_BUILD ${IE_HEADERS_DEST_DIR}/BUILD

IE_LAYER_SOURCES_DIR=$THIS_ABSOLUTE_DIR/layer_sources
IE_LAYER_SOURCES_DEST_DIR=$TF_ROOT_DIR/tensorflow/cc/inference_engine_layer

if [ -e $IE_LAYER_SOURCES_DEST_DIR ]; then
    echo "Removing old version of IE plugin source files from '$IE_LAYER_SOURCES_DEST_DIR'"
    rm -rf $IE_LAYER_SOURCES_DEST_DIR
fi
cp -r ${IE_LAYER_SOURCES_DIR} ${IE_LAYER_SOURCES_DEST_DIR}
cp $IE_LAYER_SOURCES_DEST_DIR/$LAYER_BUILD_FILE $IE_LAYER_SOURCES_DEST_DIR/BUILD

OLD_DIR=`pwd`
cd $TF_ROOT_DIR
# refer to https://github.com/allenlavoie/tensorflow/commit/4afd28316f467ac3aaf600162020637c91c0c2b7 for info about --config=monolithic
bazel build --config=monolithic //tensorflow/cc/inference_engine_layer:libtensorflow_call_layer.so
cd $OLD_DIR
