# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Define directory where artifacts will be placed
set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Search OpenVINO Inference Engine installed
find_package(OpenVINO REQUIRED)

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

add_subdirectory(common)
add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)

install(DIRECTORY scripts/ DESTINATION tests/stress_tests/scripts COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY .automation/ DESTINATION tests/stress_tests/.automation COMPONENT tests EXCLUDE_FROM_ALL)
