# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_al)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/compat_string_parser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/config/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/config/npuw.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/config/options.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/npu_mlir_runtime.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/npu_private_properties.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/npuw_private_properties.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/prefix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/profiling.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/weights_pointer_attribute.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config/npuw.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
add_library(openvino::npu_al ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME npu_al)

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        openvino::npu_logger_utils
        openvino::runtime::dev
)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET openvino::npu_al
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)
