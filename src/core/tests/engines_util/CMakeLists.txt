# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE ENGINES_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(engines_test_util STATIC EXCLUDE_FROM_ALL ${ENGINES_UTIL_SRC})

ie_faster_build(engines_test_util UNITY)

target_include_directories(engines_test_util PRIVATE $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(engines_test_util PRIVATE $<TARGET_PROPERTY:inference_engine_transformations,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(engines_test_util PUBLIC ngraph inference_engine inference_engine_transformations gtest gmock ngraph_test_util)
target_include_directories(engines_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_clang_format_target(engines_test_util_clang FOR_SOURCES ${ENGINES_UTIL_SRC})
