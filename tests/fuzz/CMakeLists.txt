# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(openvino_fuzzing)

cmake_minimum_required(VERSION 3.13)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

find_package(OpenVINODeveloperScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

# TODO: remove this when all the deprecated functions are removed
ov_deprecated_no_errors()

find_package(OpenVINO REQUIRED)

if (NOT ENABLE_FUZZING)
    message(STATUS
        "Fuzz tests will be built without fuzzer support. You can use those to\n"
        "run crash reproducers and corpus inputs. Configure ENABLE_FUZZING=ON\n"
        "to built with a fuzzer.")
else()
    enable_fuzzing()
endif()

add_subdirectory(fuzz-testhelper)
add_subdirectory(src)
