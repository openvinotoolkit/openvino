# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (APPLE)
    # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
    cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
endif()

project(InferenceEngine)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(IE_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)
include(features)

# include developer package
include(developer_package)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features)

# resolving dependencies for the project
include(dependencies)

message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "IE_MAIN_SOURCE_DIR .................... " ${IE_MAIN_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(thirdparty)

add_subdirectory(tools)

if (ENABLE_SAMPLES)
    # hint for find_package(InferenceEngine in the samples folder)
    set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")
endif()

# gflags and format_reader targets are kept inside of samples directory and
# they must be built even if samples build is disabled (required for tests and tools).
add_subdirectory(samples)

file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
add_cpplint_target(sample_cpplint
    FOR_SOURCES ${SAMPLES_SOURCES}
    EXCLUDE_PATTERNS "thirdparty/*" "pugixml/*")

if (ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

add_cpplint_report_target()
