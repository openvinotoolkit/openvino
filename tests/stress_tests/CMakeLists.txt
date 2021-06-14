# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)
set (INTTYPES_FORMAT C99)
set (BUILD_TESTING OFF)

find_package(InferenceEngineDeveloperPackage REQUIRED)

set(OpenVINO_MAIN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

if(EXISTS "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/samples/thirdparty/gflags")
    function(add_gflags)
        if(NOT WIN32)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
        endif()
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/samples/thirdparty/gflags
                         ${CMAKE_CURRENT_BINARY_DIR}/gflags_build
                         EXCLUDE_FROM_ALL)
        set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    endfunction()
    add_gflags()
endif()

add_subdirectory(unittests)
add_subdirectory(memleaks_tests)
add_subdirectory(memcheck_tests)
