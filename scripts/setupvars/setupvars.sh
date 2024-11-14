#!/bin/bash

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

abs_path () {
    script_path=$(eval echo "$1")
    directory=$(dirname "$script_path")
    builtin cd "$directory" >/dev/null 2>&1 || exit
    pwd -P
}

SCRIPT_DIR="$(abs_path "${BASH_SOURCE:-$0}")" >/dev/null 2>&1
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
    export OpenVINO_DIR=$INSTALLDIR/runtime/cmake
    # If GenAI is installed, export it as well.
    [ -f "$OpenVINO_DIR/OpenVINOGenAIConfig.cmake" ] && export OpenVINOGenAI_DIR=$OpenVINO_DIR

    system_type=$(/bin/ls "$INSTALLDIR/runtime/lib/")
    OV_PLUGINS_PATH=$INSTALLDIR/runtime/lib/$system_type

    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH=${OV_PLUGINS_PATH}/Release:${OV_PLUGINS_PATH}/Debug${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
        export LD_LIBRARY_PATH=${OV_PLUGINS_PATH}/Release:${OV_PLUGINS_PATH}/Debug${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export PKG_CONFIG_PATH=${OV_PLUGINS_PATH}/Release/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
    else
        export LD_LIBRARY_PATH=${OV_PLUGINS_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export PKG_CONFIG_PATH=$OV_PLUGINS_PATH/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
    fi

    if [ -e "$INSTALLDIR/runtime/3rdparty/tbb" ]; then
        tbb_lib_path=$INSTALLDIR/runtime/3rdparty/tbb/lib
        if [ -d "$tbb_lib_path/$system_type" ]; then
            lib_path=$(find "$tbb_lib_path/$system_type" -name "libtbb*" | sort -r | head -n1)
            if [ -n "$lib_path" ]; then
                tbb_lib_path=$(dirname "$lib_path")
            fi
        fi

        if /bin/ls "$tbb_lib_path"/libtbb* >/dev/null 2>&1; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                export DYLD_LIBRARY_PATH=$tbb_lib_path:${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
            fi
            export LD_LIBRARY_PATH=$tbb_lib_path:${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH}
        else
            echo "[setupvars.sh] WARNING: Directory with TBB libraries is not detected. Please, add TBB libraries to LD_LIBRARY_PATH / DYLD_LIBRARY_PATH manually"
        fi
        unset tbb_lib_path

        if [ -e "$INSTALLDIR/runtime/3rdparty/tbb/lib/cmake/TBB" ]; then
            export TBB_DIR=$INSTALLDIR/runtime/3rdparty/tbb/lib/cmake/TBB
        elif [ -e "$INSTALLDIR/runtime/3rdparty/tbb/lib/cmake/tbb" ]; then
            export TBB_DIR=$INSTALLDIR/runtime/3rdparty/tbb/lib/cmake/tbb
        elif [ -e "$INSTALLDIR/runtime/3rdparty/tbb/lib64/cmake/TBB" ]; then
            export TBB_DIR=$INSTALLDIR/runtime/3rdparty/tbb/lib64/cmake/TBB
        elif [ -e "$INSTALLDIR/runtime/3rdparty/tbb/cmake" ]; then
            export TBB_DIR=$INSTALLDIR/runtime/3rdparty/tbb/cmake
        else
            echo "[setupvars.sh] WARNING: TBB_DIR directory is not defined automatically by setupvars.sh. Please, set it manually to point to TBBConfig.cmake"
        fi
    fi

    unset system_type
fi

# OpenCV environment
if [ -f "$INSTALLDIR/opencv/setupvars.sh" ]; then
    # shellcheck source=/dev/null
    source "$INSTALLDIR/opencv/setupvars.sh"
fi

if [ -f "$INSTALLDIR/extras/opencv/setupvars.sh" ]; then
    # shellcheck source=/dev/null
    source "$INSTALLDIR/extras/opencv/setupvars.sh"
fi

OS_NAME=""
if command -v lsb_release >/dev/null 2>&1; then
    OS_NAME=$(lsb_release -i -s)
fi

PYTHON_VERSION_MAJOR="3"
MIN_REQUIRED_PYTHON_VERSION_MINOR="9"
MAX_SUPPORTED_PYTHON_VERSION_MINOR="13"

check_python_version () {
    if [ -z "$python_version" ]; then
        python_version_major=$( python3 -c 'import sys; print(str(sys.version_info[0]))' )
        python_version_minor=$( python3 -c 'import sys; print(str(sys.version_info[1]))' )
        python_version="$python_version_major.$python_version_minor"
    else
        python_version_major=$( python3 -c "import sys; print(str(\"${python_version}\".split('.')[0]))" )
        python_version_minor=$( python3 -c "import sys; print(str(\"${python_version}\".split('.')[1]))" )
    fi

    if  [ "$PYTHON_VERSION_MAJOR" != "$python_version_major" ] ||
        [ "$python_version_minor" -lt "$MIN_REQUIRED_PYTHON_VERSION_MINOR" ] ||
        [ "$python_version_minor" -gt "$MAX_SUPPORTED_PYTHON_VERSION_MINOR" ] ; then
        echo "[setupvars.sh] WARNING: Unsupported Python version ${python_version}. Please install one of Python" \
        "${PYTHON_VERSION_MAJOR}.${MIN_REQUIRED_PYTHON_VERSION_MINOR} -" \
        "${PYTHON_VERSION_MAJOR}.${MAX_SUPPORTED_PYTHON_VERSION_MINOR} (64-bit) from https://www.python.org/downloads/"
        unset python_version
        return 0
    fi

    if command -v python"$python_version" > /dev/null 2>&1; then
        python_interp=python"$python_version"
    else
        python_interp=python"$python_version_major"
    fi
    python_bitness=$("$python_interp" -c 'import sys; print(64 if sys.maxsize > 2**32 else 32)')
    unset python_interp

    if [ "$python_bitness" != "" ] && [ "$python_bitness" != "64" ] && [ "$OS_NAME" != "Raspbian" ]; then
        echo "[setupvars.sh] WARNING: 64 bitness for Python $python_version is required"
    fi
    unset python_bitness

    if [ -n "$python_version" ]; then
        if [[ -d $INTEL_OPENVINO_DIR/python ]]; then
            # add path to OpenCV API for Python 3.x
            export PYTHONPATH="$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH"
            # add path to OpenVINO Python API
            export PYTHONPATH="$INTEL_OPENVINO_DIR/python:${PYTHONPATH}"
        else
            echo "[setupvars.sh] WARNING: Can not find OpenVINO Python binaries by path ${INTEL_OPENVINO_DIR}/python"
            echo "[setupvars.sh] WARNING: OpenVINO Python environment does not set properly"
        fi
    fi
}

python_version_to_check="$python_version"
if [ -z "$python_version" ]; then
    python_version_to_check="3"
fi

if ! command -v python"$python_version_to_check" > /dev/null 2>&1; then
    echo "[setupvars.sh] WARNING: Python is not installed. Please install one of Python" \
    "${PYTHON_VERSION_MAJOR}.${MIN_REQUIRED_PYTHON_VERSION_MINOR} -" \
    "${PYTHON_VERSION_MAJOR}.${MAX_SUPPORTED_PYTHON_VERSION_MINOR} (64-bit) from https://www.python.org/downloads/"
else
    check_python_version
fi

unset python_version
unset python_version_to_check
unset PYTHON_VERSION_MAJOR
unset MIN_REQUIRED_PYTHON_VERSION_MINOR
unset MAX_SUPPORTED_PYTHON_VERSION_MINOR
unset OS_NAME

echo "[setupvars.sh] OpenVINO environment initialized"
