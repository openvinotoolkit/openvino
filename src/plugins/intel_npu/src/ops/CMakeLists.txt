# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_intel_npu_ops)
set(TARGET_ALIAS openvino::intel_npu::ops)
set(TARGET_EXPORT_NAME intel_npu::ops)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(${TARGET_NAME} STATIC ${SOURCES})
add_library(${TARGET_ALIAS} ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES 
    EXPORT_NAME ${TARGET_EXPORT_NAME}
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::reference
    openvino::conditional_compilation
    openvino::itt
    openvino::core::dev
    openvino::shape_inference
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

#
# targets install
#

ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})
ov_developer_package_export_targets(TARGET ${TARGET_NAME})
