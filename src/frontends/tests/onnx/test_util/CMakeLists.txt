# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE ONNX_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(onnx_test_util STATIC EXCLUDE_FROM_ALL ${ONNX_UTIL_SRC})

ie_faster_build(onnx_test_util UNITY)

target_link_libraries(onnx_test_util PRIVATE onnx_common)
target_include_directories(onnx_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_clang_format_target(onnx_test_util_clang FOR_SOURCES ${ONNX_UTIL_SRC})
