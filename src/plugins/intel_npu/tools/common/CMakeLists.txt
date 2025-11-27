#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_tools_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL
    ${SOURCES} "${NPU_PLUGIN_SOURCE_DIR}/src/plugin/src/metadata.cpp")

set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
                          CXX_STANDARD 17)

target_include_directories(${TARGET_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${NPU_PLUGIN_SOURCE_DIR}/src/plugin/include>")

target_link_libraries(${TARGET_NAME} PUBLIC
    openvino::core::dev
    openvino::runtime
    openvino_npu_logger_utils)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()
