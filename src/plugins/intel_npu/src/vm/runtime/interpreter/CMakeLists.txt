#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "npu_host_interpreter")

file(GLOB_RECURSE SOURCES *.cpp *.hpp)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::npu_al
        openvino::npu_zero_utils
        # VPUXLoader
        level_zero_wrapper
	    openvino_npu_logger_utils
        # npu_elf

        openvino::npu_mlir_runtime_api
        openvino::npu_bytecode
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${PLUGIN_COMPILER_LIB_PATH}/../vpux_elf/loader/include
)
