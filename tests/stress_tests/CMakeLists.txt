# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_DOCKER)
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
else()
    if (APPLE)
        # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
        cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
    else()
        cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
    endif()
endif()

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(InferenceEngineDeveloperPackage REQUIRED)

add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)
