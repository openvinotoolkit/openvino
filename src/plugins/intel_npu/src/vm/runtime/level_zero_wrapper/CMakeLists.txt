#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME level_zero_wrapper)

add_library(${TARGET_NAME} SHARED
    level_zero_wrapper.h
    level_zero_wrapper.cpp
)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<TARGET_PROPERTY:LevelZero::Headers,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:LevelZero::NPUExt,INTERFACE_INCLUDE_DIRECTORIES>) 

add_library(VPUXLoader STATIC IMPORTED)

set_target_properties(VPUXLoader PROPERTIES
    IMPORTED_LOCATION "${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/lib/Release/libVPUXLoader.a"
    INTERFACE_INCLUDE_DIRECTORIES 
        "${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/loader/include;${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/core/include"
)

add_library(npu_elf STATIC IMPORTED)
set_target_properties(npu_elf PROPERTIES
    IMPORTED_LOCATION "${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/lib/Release/libnpu_elf.a"
    INTERFACE_INCLUDE_DIRECTORIES 
        "${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/loader/include;${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/core/include"
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    npu_elf
    VPUXLoader
    openvino::npu_al
    openvino::npu_bytecode
    openvino::ze_loader
    openvino::npu_zero_utils)

# target_include_directories(${TARGET_NAME}
#     PRIVATE
#         ${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/loader/include
# )
