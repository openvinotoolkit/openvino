# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
project(llmdnn_tests)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(googletest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include_directories(${LLMDNN_HEADERS_DIR})

file(GLOB_RECURSE TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(llmdnn_tests ${TEST_SOURCE_FILES})
target_link_libraries(llmdnn_tests llmdnn gtest_main stdc++)
install(TARGETS llmdnn_tests DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
