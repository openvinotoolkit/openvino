#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_tools_utils")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/data_type_converters.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tensor_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_utils.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES
                          FOLDER intel_npu_tools
                          CXX_STANDARD 17)

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()
