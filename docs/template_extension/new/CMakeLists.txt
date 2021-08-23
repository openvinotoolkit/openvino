# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "template_ov_extension")

find_package(ngraph REQUIRED)

set(SRC ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE template_operations
                                             ngraph::ngraph)
# [cmake:extension]

# Enable code style check
file(GLOB_RECURSE template_ov_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${template_ov_extension_src})
