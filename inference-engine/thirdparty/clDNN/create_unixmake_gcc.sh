# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

BOOST_VERSION="1.64.0"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${ROOT_DIR}/build/Linux64"
OUT_DIR="${ROOT_DIR}/build/out/Linux64"

USE_NINJA=${1:-"N"}
USE_DEVTOOLSET="${2}"

if [ "${USE_NINJA^^}" = "Y" ]; then
    echo "Creating Ninja Makefiles ..."
    GENERATOR="Ninja"
else
    echo "Creating Unix/Linux Makefiles ..."
    GENERATOR="Unix Makefiles"
fi

if [ "${USE_DEVTOOLSET}" = "" ]; then
    cd ${ROOT_DIR} && cmake -E make_directory "${BUILD_DIR}/Debug" && cd "${BUILD_DIR}/Debug" && cmake -G "${GENERATOR}" "-DCLDNN__OUTPUT_DIR=${OUT_DIR}/Debug" "-DCMAKE_BUILD_TYPE=Debug" "-DCLDNN__BOOST_VERSION=${BOOST_VERSION}" "${ROOT_DIR}"
    cd ${ROOT_DIR} && cmake -E make_directory "${BUILD_DIR}/Release" && cd "${BUILD_DIR}/Release" && cmake -G "${GENERATOR}" "-DCLDNN__OUTPUT_DIR=${OUT_DIR}/Release" "-DCMAKE_BUILD_TYPE=Release" "-DCLDNN__BOOST_VERSION=${BOOST_VERSION}" "${ROOT_DIR}"
else
    echo Using devtoolset-${USE_DEVTOOLSET,,} ...
    cd ${ROOT_DIR} && cmake -E make_directory "${BUILD_DIR}/Debug" && cd "${BUILD_DIR}/Debug" && scl enable devtoolset-${USE_DEVTOOLSET,,} "cmake -G \"${GENERATOR}\" \"-DCLDNN__OUTPUT_DIR=${OUT_DIR}/Debug\" \"-DCMAKE_BUILD_TYPE=Debug\" \"-DCLDNN__BOOST_VERSION=${BOOST_VERSION}\" \"${ROOT_DIR}\""
    cd ${ROOT_DIR} && cmake -E make_directory "${BUILD_DIR}/Release" && cd "${BUILD_DIR}/Release" && scl enable devtoolset-${USE_DEVTOOLSET,,} "cmake -G \"${GENERATOR}\" \"-DCLDNN__OUTPUT_DIR=${OUT_DIR}/Release\" \"-DCMAKE_BUILD_TYPE=Release\" \"-DCLDNN__BOOST_VERSION=${BOOST_VERSION}\" \"${ROOT_DIR}\""
fi


echo Done.
