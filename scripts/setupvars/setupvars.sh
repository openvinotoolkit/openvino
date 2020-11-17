#!/bin/bash

# Copyright (c) 2018-2020 Intel Corporation
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="$( dirname "$SCRIPT_DIR" )"

INSTALLDIR="${BASE_DIR}"


export INTEL_OPENVINO_DIR="$INSTALLDIR"
export INTEL_CVSDK_DIR="$INTEL_OPENVINO_DIR"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -pyver)
    python_version=$2
    echo python_version = "${python_version}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

if [ -e "$INSTALLDIR/deployment_tools/inference_engine" ]; then
    export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
    system_type=$(ls "$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/")
    IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/$system_type

    if [[ -e ${IE_PLUGINS_PATH}/arch_descriptions ]]; then
        export ARCH_ROOT_DIR=${IE_PLUGINS_PATH}/arch_descriptions
    fi

    export HDDL_INSTALL_DIR=$INSTALLDIR/deployment_tools/inference_engine/external/hddl
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_mac/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:${IE_PLUGINS_PATH}${DYLD_LIBRARY_PATH:+:DYLD_LIBRARY_PATH}
        export LD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_mac/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:${IE_PLUGINS_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    else
        export LD_LIBRARY_PATH=$HDDL_INSTALL_DIR/lib:$INSTALLDIR/deployment_tools/inference_engine/external/gna/lib:$INSTALLDIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INSTALLDIR/deployment_tools/inference_engine/external/tbb/lib:${IE_PLUGINS_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi

    HDDL_UNITE_DIR=$INSTALLDIR/deployment_tools/inference_engine/external/hddl_unite

    if [ -e "$HDDL_UNITE_DIR" ]; then
        export LD_LIBRARY_PATH=$HDDL_UNITE_DIR/lib:$HDDL_UNITE_DIR/thirdparty/XLink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi
fi

if [ -e "$INSTALLDIR/deployment_tools/ngraph" ]; then
    export LD_LIBRARY_PATH=$INSTALLDIR/deployment_tools/ngraph/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export ngraph_DIR=$INSTALLDIR/deployment_tools/ngraph/cmake
fi

if [ -e "$INSTALLDIR/opencv" ]; then
    if [ -f "$INSTALLDIR/opencv/setupvars.sh" ]; then
        source "$INSTALLDIR/opencv/setupvars.sh"
    else
        export OpenCV_DIR="$INSTALLDIR/opencv/share/OpenCV"
        export LD_LIBRARY_PATH="$INSTALLDIR/opencv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LD_LIBRARY_PATH="$INSTALLDIR/opencv/share/OpenCV/3rdparty/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi


if [ -f "$INTEL_OPENVINO_DIR/data_processing/dl_streamer/bin/setupvars.sh" ]; then
    source "$INTEL_OPENVINO_DIR/data_processing/dl_streamer/bin/setupvars.sh"
fi

export PATH="$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer${PATH:+:$PATH}"
export PYTHONPATH="$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer${PYTHONPATH:+:$PYTHONPATH}"


if [ -e "$INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/accuracy_checker" ]; then
    export PYTHONPATH="$INTEL_OPENVINO_DIR/deployment_tools/open_model_zoo/tools/accuracy_checker:$PYTHONPATH"
fi

if [ -e "$INTEL_OPENVINO_DIR/deployment_tools/tools/post_training_optimization_toolkit" ]; then
    export PYTHONPATH="$INTEL_OPENVINO_DIR/deployment_tools/tools/post_training_optimization_toolkit:$PYTHONPATH"
fi

if [ -z "$python_version" ]; then
    python_version=$(python3 -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')
fi

OS_NAME=""
if command -v lsb_release >/dev/null 2>&1; then
    OS_NAME=$(lsb_release -i -s)
fi

python_bitness=$(python3 -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')   
if [ "$python_bitness" != "" ] && [ "$python_bitness" != "64" ] && [ "$OS_NAME" != "Raspbian" ]; then
    echo "[setupvars.sh] 64 bitness for Python $python_version is required"
fi

MINIMUM_REQUIRED_PYTHON_VERSION="3.6"
MAX_SUPPORTED_PYTHON_VERSION=$([[ "$OSTYPE" == "darwin"* ]] && echo '3.7' || echo '3.8') 
if [[ -n "$python_version" && "$(printf '%s\n' "$python_version" "$MINIMUM_REQUIRED_PYTHON_VERSION" | sort -V | head -n 1)" != "$MINIMUM_REQUIRED_PYTHON_VERSION" ]]; then
    echo "[setupvars.sh] ERROR: Unsupported Python version. Please install one of Python 3.6-${MAX_SUPPORTED_PYTHON_VERSION} (64-bit) from https://www.python.org/downloads/"
    return 1
fi


if [ -n "$python_version" ]; then
    if [[ -d $INTEL_OPENVINO_DIR/python ]]; then
        # add path to OpenCV API for Python 3.x
        export PYTHONPATH="$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH"
        pydir=$INTEL_OPENVINO_DIR/python/python$python_version
        if [[ -d $pydir ]]; then
            # add path to Inference Engine Python API
            export PYTHONPATH="${pydir}:${PYTHONPATH}"
        else
            echo "[setupvars.sh] WARNING: Can not find OpenVINO Python module for python${python_version} by path ${pydir}"
            echo "[setupvars.sh] WARNING: OpenVINO Python environment does not set properly"
        fi
    else
        echo "[setupvars.sh] WARNING: Can not find OpenVINO Python binaries by path ${INTEL_OPENVINO_DIR}/python"
        echo "[setupvars.sh] WARNING: OpenVINO Python environment does not set properly"
    fi
fi

echo "[setupvars.sh] OpenVINO environment initialized"
