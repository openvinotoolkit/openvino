#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"
INSTALLDIR="${SCRIPT_DIR}"
export INTEL_OPENVINO_DIR="$INSTALLDIR"

# parse command line options
while [ $# -gt 0 ]
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

if [ -e "$INSTALLDIR/runtime" ]; then
    export InferenceEngine_DIR=$INSTALLDIR/runtime/cmake
    export ngraph_DIR=$INSTALLDIR/runtime/cmake
    export OpenVINO_DIR=$INSTALLDIR/runtime/cmake

    system_type=$(ls "$INSTALLDIR/runtime/lib/")
    IE_PLUGINS_PATH=$INSTALLDIR/runtime/lib/$system_type

    export HDDL_INSTALL_DIR=$INSTALLDIR/runtime/3rdparty/hddl
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH=${IE_PLUGINS_PATH}/Release:${IE_PLUGINS_PATH}/Debug${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
        export LD_LIBRARY_PATH=${IE_PLUGINS_PATH}/Release:${IE_PLUGINS_PATH}/Debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    else
        export LD_LIBRARY_PATH=$HDDL_INSTALL_DIR/lib:${IE_PLUGINS_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi

    HDDL_UNITE_DIR=$INSTALLDIR/runtime/3rdparty/hddl_unite

    if [ -e "$HDDL_UNITE_DIR" ]; then
        export LD_LIBRARY_PATH=$HDDL_UNITE_DIR/lib:$HDDL_UNITE_DIR/thirdparty/XLink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    fi
fi

if [ -e "$INSTALLDIR/runtime/3rdparty/tbb" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH=$INSTALLDIR/runtime/3rdparty/tbb/lib:${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
    fi
    export LD_LIBRARY_PATH=$INSTALLDIR/runtime/3rdparty/tbb/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export TBB_DIR=$INSTALLDIR/runtime/3rdparty/tbb/cmake
fi

if [ -e "$INSTALLDIR/tools/compile_tool" ]; then
    export LD_LIBRARY_PATH=$INSTALLDIR/tools/compile_tool${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# OpenCV environment
if [ -f "$INSTALLDIR/opencv/setupvars.sh" ]; then
    source "$INSTALLDIR/opencv/setupvars.sh"
fi

if [ -f "$INSTALLDIR/extras/opencv/setupvars.sh" ]; then
    source "$INSTALLDIR/extras/opencv/setupvars.sh"
fi

if [ -z "$python_version" ]; then
    python_version=$(python3 -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')
fi

# splitting Python version variable depending on the used shell 
if [ -n "$ZSH_VERSION" ]; then
    version_arr=(${(@s:.:)python_version})
    if [ "${#version_arr[@]}" -ge "2" ]; then
        # zsh starts indexing from 1
        python_version_major=${version_arr[1]}
        python_version_minor=${version_arr[2]}
    fi
else
    version_arr=(${python_version//./ })
    if [ "${#version_arr[@]}" -ge "2" ]; then
        python_version_major=${version_arr[0]}
        python_version_minor=${version_arr[1]}
    fi
fi

PYTHON_VERSION_MAJOR="3"
MIN_REQUIRED_PYTHON_VERSION_MINOR="6"
MAX_SUPPORTED_PYTHON_VERSION_MINOR="9"

if  [ "$PYTHON_VERSION_MAJOR" != "$python_version_major" ] ||
    [ "$python_version_minor" -lt "$MIN_REQUIRED_PYTHON_VERSION_MINOR" ] ||
    [ "$python_version_minor" -gt "$MAX_SUPPORTED_PYTHON_VERSION_MINOR" ] ; then
    echo "[setupvars.sh] ERROR: Unsupported Python version. Please install one of Python" \
    "${PYTHON_VERSION_MAJOR}.${MIN_REQUIRED_PYTHON_VERSION_MINOR} -" \
    "${PYTHON_VERSION_MAJOR}.${MAX_SUPPORTED_PYTHON_VERSION_MINOR} (64-bit) from https://www.python.org/downloads/"
    return 1
fi

OS_NAME=""
if command -v lsb_release >/dev/null 2>&1; then
    OS_NAME=$(lsb_release -i -s)
fi

python_bitness=$(python"$python_version" -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
if [ "$python_bitness" != "" ] && [ "$python_bitness" != "64" ] && [ "$OS_NAME" != "Raspbian" ]; then
    echo "[setupvars.sh] WARNING: 64 bitness for Python $python_version is required"
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
