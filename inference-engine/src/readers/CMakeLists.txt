# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME inference_engine_reader_api)

# Reader API interface library
add_library(${TARGET_NAME} INTERFACE)

target_include_directories(${TARGET_NAME} INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/reader_api"
    $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>)

file(GLOB_RECURSE reader_api_hpp "${CMAKE_CURRENT_SOURCE_DIR}/reader_api/*.hpp")
file(GLOB_RECURSE reader_api_src "${CMAKE_CURRENT_SOURCE_DIR}/reader_api/*.cpp")

set_target_properties(${TARGET_NAME} PROPERTIES INTERFACE_SOURCES ${reader_api_src})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_SOURCES ${reader_api_hpp} ${reader_api_src})

add_subdirectory(ir_reader)
add_subdirectory(ir_reader_v7)

if(NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx_reader)
endif()
