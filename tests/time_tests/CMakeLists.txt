# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(InferenceEngineDeveloperPackage QUIET)
if (NOT InferenceEngineDeveloperPackage_FOUND)
    find_package(InferenceEngine REQUIRED)
endif()

add_subdirectory(src)
