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

find_package(InferenceEngineDeveloperPackage REQUIRED)

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

add_subdirectory(common)
add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)
