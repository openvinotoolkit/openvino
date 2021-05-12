# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)
set (INTTYPES_FORMAT C99)
find_package(InferenceEngineDeveloperPackage REQUIRED)

add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)
