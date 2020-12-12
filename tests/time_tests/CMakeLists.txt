# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")

project(time_tests)


find_package(InferenceEngine)
if (NOT InferenceEngine_FOUND)
    set (HAVE_SYS_STAT_H 1)
    set (HAVE_INTTYPES_H 1)
    set (INTTYPES_FORMAT C99)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

add_subdirectory(src)
