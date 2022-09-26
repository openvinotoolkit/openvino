#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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


V_ENV=0

for ((i=1;i <= $#;i++)) {
    case "${!i}" in
        caffe|tf|tf2|mxnet|kaldi|onnx)
            postfix="_$1"
            ;;
        "venv")
            V_ENV=1
            ;;
        *)
            if [[ "$1" != "" ]]; then
                echo "\"${!i}\" is unsupported parameter"
                echo $"Usage: $0 {caffe|tf|tf2|mxnet|kaldi|onnx} {venv}"
                exit 1
            fi
            ;;
        esac
}

VENV_DIR="$HOME/venv_openvino"
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" && pwd )"

if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
fi

if [[ $DISTRO == "centos" ]]; then
    if command -v python3.8 >/dev/null 2>&1; then
        python_binary=python3.8
    elif command -v python3.7 >/dev/null 2>&1; then
        python_binary=python3.7
    elif command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    fi
else
    python_binary=python3
fi

install_latest_ov() {
    if $2; then
        sudo -E "$1" -m pip install openvino
    else
        "$1" -m pip install openvino
    fi
}

install_ov() {
    if $2; then
        sudo -E "$1" -m pip install openvino=="$3"
    else
        "$1" -m pip install openvino=="$3"
    fi
}

uninstall_ov() {
    if $2; then
        sudo -E "$1" -m pip uninstall -y openvino
    else
        "$1" -m pip uninstall -y openvino
    fi
}

check_ie() {
    $1 "$SCRIPTDIR/../openvino/tools/mo/utils/find_ie_version.py"
}

check_ov_package() {
    if $2; then
        sudo -E "$1" -m pip show openvino
    else
        "$1" -m pip show openvino
    fi
}

print_warning() {
    YELLOW='\033[1;33m'
    NC='\033[0m'
    printf "${YELLOW}[ WARNING ] %s ${NC}\n" "$1"
}

find_ie_bindings() {
    python_executable="$1"
    requires_sudo="$2"

    mo_release_version="$("$python_executable" "$SCRIPTDIR"/../openvino/tools/mo/utils/extract_release_version.py)"
    if [[ $mo_release_version == "None.None" ]]; then
      mo_is_custom=true
    else
      mo_is_custom=false
    fi

    if ! check_ie "$python_executable"; then
        # Check if OpenVINO version was installed using pip
        if check_ov_package "$python_executable" "$requires_sudo"; then
            if $mo_is_custom; then
                print_warning "OpenVINO (TM) Toolkit version installed in pip is incompatible with the Model Optimizer."
                print_warning "For the custom Model Optimizer version consider building Inference Engine Python API from sources (preferable) or install the highest OpenVINO (TM) toolkit version using \"pip install openvino\""
            else
                print_warning "OpenVINO (TM) Toolkit version installed in pip is incompatible with the Model Optimizer."
                print_warning "For the release version of the Model Optimizer, which is $mo_release_version, install the OpenVINO (TM) toolkit using \"pip install openvino==$mo_release_version\" or build the Inference Engine Python API from sources."
            fi
            return 0
        fi

        print_warning "Could not find the Inference Engine Python API. Installing OpenVINO (TM) toolkit using pip."

        if $mo_is_custom; then
            print_warning "Detected a custom Model Optimizer version."
            print_warning "The desired version of the Inference Engine can be installed only for the release Model Optimizer version."
            print_warning "The highest OpenVINO (TM) toolkit version will be installed, which might be incompatible with the current Model Optimizer version."
            print_warning "It is recommended to build the Inference Engine from sources even if the current installation is successful."
        elif install_ov "$python_executable" "$requires_sudo" "$mo_release_version"; then
            if check_ie "$python_executable"; then
                return 0
            fi

            print_warning "The installed OpenVINO (TM) toolkit version $mo_release_version does not work as expected. Uninstalling..."
            uninstall_ov "$python_executable" "$requires_sudo"
            print_warning "Consider building the Inference Engine Python API from sources."
            return 0
        else
            print_warning "Could not find the OpenVINO (TM) toolkit version $mo_release_version in pip."
            print_warning "The highest OpenVINO (TM) toolkit version will be installed, which might be incompatible with the current Model Optimizer version."
            print_warning "It is recommended to build the Inference Engine from sources even if the current installation is successful."
        fi

        # Install the highest OpenVINO pip version
        if install_latest_ov "$python_executable" "$requires_sudo"; then
            if check_ie "$python_executable"; then
                return 0
            else
                print_warning "The installed highest OpenVINO (TM) toolkit version doesn't work as expected. Uninstalling..."
                uninstall_ov "$python_executable" "$requires_sudo"
                print_warning "Consider building the Inference Engine Python API from sources."
                return 0
            fi
        else
            print_warning "Could not find OpenVINO (TM) toolkit version available in pip for installation."
            print_warning "Consider building the Inference Engine Python API from sources."
            return 0
        fi
    fi

    return 0
}

if [[ $V_ENV -eq 1 ]]; then
    "$python_binary" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    venv_python_binary="$VENV_DIR/bin/$python_binary"
    # latest pip is needed to install tensorflow
    "$venv_python_binary" -m pip install --upgrade pip
    "$venv_python_binary" -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
    find_ie_bindings "$venv_python_binary" false
    echo
    echo "Before running the Model Optimizer, please activate virtualenv environment by running \"source $VENV_DIR/bin/activate\""
else
    # latest pip is needed to install tensorflow
    "$python_binary" -m pip install --upgrade pip
    "$python_binary" -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
    find_ie_bindings "$python_binary" false
    echo
    echo "[WARNING] All Model Optimizer dependencies are installed globally."
    echo "[WARNING] If you want to keep Model Optimizer in separate sandbox"
    echo "[WARNING] run install_prerequisites.sh \"{caffe|tf|tf2|mxnet|kaldi|onnx}\" venv"
fi
