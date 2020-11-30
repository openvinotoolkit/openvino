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

cmake_minimum_required (VERSION 3.13)

project (pyngraph)

include(ExternalProject)

ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.5.0"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

set(BUILD_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.py.in")
set(BUILD_PY    "${CMAKE_CURRENT_BINARY_DIR}/build_wheel.py")
configure_file(${BUILD_PY_IN} ${BUILD_PY} @ONLY)

add_custom_command(
        DEPENDS pybind11
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist/
        POST_BUILD
        WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND python ${BUILD_PY}
)

add_custom_target(python_wheel DEPENDS ngraph ${CMAKE_CURRENT_BINARY_DIR}/dist/)
