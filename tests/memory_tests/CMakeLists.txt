# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Define directory where artifacts will be placed
set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

project(memory_tests)

find_package(InferenceEngine REQUIRED)

add_subdirectory(src)
