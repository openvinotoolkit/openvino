# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")

project(time_tests)

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

# Search OpenVINO Inference Engine installed
find_package(InferenceEngine)

if(NOT InferenceEngine_FOUND)
    # Define directory where artifacts will be placed
    set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

    # Search OpenVINO Inference Engine via InferenceEngineDeveloperPackage_DIR
    # in order to provide backward compatibility with old OpenVINO packages
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

add_subdirectory(src)
