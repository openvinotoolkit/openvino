#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npu_vm_runtime_api)

add_library(${TARGET_NAME} INTERFACE)
add_library(openvino::npu_vm_runtime_api ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# prepare header dependencies
target_link_libraries(${TARGET_NAME} INTERFACE
    LevelZero::NPUExt     # contain ze_graph_ext.h
    openvino::zero_loader # contain level_zero/ze_api.h>
)

# prepare file for build
if(TARGET prepare_ze_headers)
    add_dependencies(${TARGET_NAME} prepare_ze_headers)
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_developer_package_export_targets(TARGET openvino::npu_vm_runtime_api
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)
ov_install_static_lib(npu_vm_runtime_api ${NPU_PLUGIN_COMPONENT})
