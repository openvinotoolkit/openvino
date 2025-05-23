# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_common)

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
add_library(openvino::npu_common ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME npu_common)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        openvino::npu_al
        openvino::npu_logger_utils
        openvino::npu_zero_utils
)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET openvino::npu_common
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include)
