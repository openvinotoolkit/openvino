#!/bin/bash

# Copyright (C) 2018-2021 Intel Corporation
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

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
    elif command -v python3.5 >/dev/null 2>&1; then
        python_binary=python3.5
    fi

    if [ -z "$python_binary" ]; then
        sudo -E yum install -y https://centos7.iuscommunity.org/ius-release.rpm
        sudo -E yum install -y python36u python36u-pip
        sudo -E pip3.6 install virtualenv
        python_binary=python3.6
    fi
    # latest pip is needed to install tensorflow
    sudo -E "$python_binary" -m pip install --upgrade pip
elif [[ $DISTRO == "ubuntu" ]]; then
    sudo -E apt update
    sudo -E apt -y --no-install-recommends install python3-pip python3-venv
    python_binary=python3
    sudo -E "$python_binary" -m pip install --upgrade pip
elif [[ "$OSTYPE" == "darwin"* ]]; then
    python_binary=python3
    python3 -m pip install --upgrade pip
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
    $1 "$SCRIPTDIR/../mo/utils/find_ie_version.py"
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

    mo_release_version="$("$python_executable" "$SCRIPTDIR"/../mo/utils/extract_release_version.py)"
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
    "$python_binary" -m venv "$SCRIPTDIR/../venv${postfix}"
    source "$SCRIPTDIR/../venv${postfix}/bin/activate"
    venv_python_binary="$SCRIPTDIR/../venv${postfix}/bin/$python_binary"
    $venv_python_binary -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
    find_ie_bindings "$venv_python_binary" false
    echo
    echo "Before running the Model Optimizer, please activate virtualenv environment by running \"source ${SCRIPTDIR}/../venv${postfix}/bin/activate\""
else
    if [[ "$OSTYPE" == "darwin"* ]]; then
        python3 -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
        find_ie_bindings python3 false
    else
        sudo -E $python_binary -m pip install -r "$SCRIPTDIR/../requirements${postfix}.txt"
        find_ie_bindings $python_binary true
    fi
    echo "[WARNING] All Model Optimizer dependencies are installed globally."
    echo "[WARNING] If you want to keep Model Optimizer in separate sandbox"
    echo "[WARNING] run install_prerequisites.sh \"{caffe|tf|tf2|mxnet|kaldi|onnx}\" venv"
fi
