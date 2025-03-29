# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(WIN32)
    cmake_minimum_required(VERSION 3.16)
else()
    cmake_minimum_required(VERSION 3.13)
endif()

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()

set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")

project(time_tests)

set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

add_subdirectory(src)

install(DIRECTORY test_runner/ DESTINATION tests/time_tests/test_runner COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY .automation/ DESTINATION tests/time_tests/test_runner/.automation COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY scripts/ DESTINATION tests/time_tests/scripts COMPONENT tests EXCLUDE_FROM_ALL)
install(DIRECTORY ../utils/ DESTINATION tests/utils COMPONENT tests EXCLUDE_FROM_ALL)
