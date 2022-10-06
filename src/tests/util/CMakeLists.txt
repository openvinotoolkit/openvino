# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4101)
    ie_add_compiler_flags(/wd4267)
endif()

add_library(ngraph_test_util STATIC EXCLUDE_FROM_ALL ${UTIL_SRC})

ie_faster_build(ngraph_test_util UNITY)

target_link_libraries(ngraph_test_util PUBLIC openvino::runtime openvino::core::dev gtest gmock)
target_include_directories(ngraph_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ngraph_test_util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

file(GLOB_RECURSE all_util_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(ngraph_test_util_clang FOR_SOURCES ${all_util_src})

# developer package

openvino_developer_export_targets(COMPONENT tests TARGETS ngraph_test_util)
