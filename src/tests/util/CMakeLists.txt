# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(ngraph_test_util STATIC EXCLUDE_FROM_ALL ${UTIL_SRC})
target_compile_definitions(ngraph_test_util PRIVATE

    # Workaround, c++17 introduce std::byte which can colide with byte defined in MSVS header
    # related to test_common.cpp
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:$<$<COMPILE_FEATURES:cxx_std_17>:_HAS_STD_BYTE=0>>)
ie_faster_build(ngraph_test_util UNITY)

target_link_libraries(ngraph_test_util PUBLIC openvino::runtime gtest gmock)
target_include_directories(ngraph_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ngraph_test_util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

file(GLOB_RECURSE all_util_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(ngraph_test_util_clang FOR_SOURCES ${all_util_src})

# developer package

openvino_developer_export_targets(COMPONENT tests TARGETS ngraph_test_util)
