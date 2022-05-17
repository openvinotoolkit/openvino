#! /bin/bash

# *******************************************************************************
# Copyright 2020-2021 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# *******************************************************************************

# Compute Library build defaults
ACL_VERSION="v21.08"
ACL_DIR="${PWD}/ComputeLibrary"
ACL_ARCH="arm64-v8a"

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
        ACL_VERSION="v$2"
        shift
        ;;
        --arch)
        ACL_ARCH="$2"
        shift
        ;;
        --root-dir)
        ACL_DIR="$2"
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

readonly ACL_REPO="https://github.com/ARM-software/ComputeLibrary.git"
MAKE_NP="-j$(grep -c processor /proc/cpuinfo)"

git clone $ACL_REPO $ACL_DIR
cd $ACL_DIR
git checkout $ACL_VERSION

scons $MAKE_NP Werror=0 debug=0 neon=1 gles_compute=0 embed_kernels=0 \
  os=linux arch=$ACL_ARCH build=native

exit $?
