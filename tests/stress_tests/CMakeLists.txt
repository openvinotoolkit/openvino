# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(stress_tests)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

# Define directory where artifacts will be placed
set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

find_package(OpenVINODeveloperScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

add_subdirectory(common)
add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)

install(DIRECTORY scripts/ DESTINATION tests/stress_tests/scripts COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY .automation/ DESTINATION tests/stress_tests/.automation COMPONENT tests EXCLUDE_FROM_ALL)
