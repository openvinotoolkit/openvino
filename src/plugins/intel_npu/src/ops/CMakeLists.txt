# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_ops)
set(TARGET_ALIAS openvino::npu_ops)
set(TARGET_EXPORT_NAME npu_ops)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

add_library(${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})
add_library(${TARGET_ALIAS} ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(ENABLE_MLIR_COMPILER)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_MLIR_COMPILER)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    openvino::reference
    openvino::conditional_compilation
    openvino::core::dev
    openvino::shape_inference
)

set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME ${TARGET_EXPORT_NAME}
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

#
# targets install
#

ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET ${TARGET_ALIAS}
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)
