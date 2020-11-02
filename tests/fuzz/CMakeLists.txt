# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(CMAKE_MODULE_PATH "${OpenVINO_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT TARGET IE::inference_engine)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

include(sanitizer)
include(fuzzing)

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
