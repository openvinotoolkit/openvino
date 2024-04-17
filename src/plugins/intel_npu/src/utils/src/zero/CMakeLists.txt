# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_zero_result_parser)

file(GLOB_RECURSE SOURCES *.cpp)

add_library(${TARGET_NAME} STATIC ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${NPU_PLUGIN_SOURCE_DIR}/thirdparty/level-zero/include>
        $<BUILD_INTERFACE:${NPU_PLUGIN_SOURCE_DIR}/thirdparty/level-zero-ext>
        $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>
)

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})
