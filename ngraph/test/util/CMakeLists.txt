# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
if (NOT NGRAPH_ONNX_FRONTEND_ENABLE)
    list(REMOVE_ITEM UTIL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/onnx_test_util.cpp ${CMAKE_CURRENT_SOURCE_DIR}/onnx_test_util.hpp)
endif()

add_library(ngraph_test_util STATIC ${UTIL_SRC})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph_test_util UNITY)
endif()

target_link_libraries(ngraph_test_util PUBLIC ngraph ngraph_backend gtest gmock)
if (NGRAPH_ONNX_FRONTEND_ENABLE)
    target_link_libraries(ngraph_test_util PRIVATE onnx_common)
endif()
target_include_directories(ngraph_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE all_util_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(ngraph_test_util_clang FOR_SOURCES ${all_util_src})

# developer package
openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_test_util)
