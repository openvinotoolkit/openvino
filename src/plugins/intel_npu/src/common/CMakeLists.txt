# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_common")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/compiler_adapter_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/option_support_cache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/device_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/icompiled_model.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/icontext.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/idevice.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/idriver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iengine_backend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iinfer_request_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/infer_request.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/icompiler_adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/igraph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iprofiling_decoder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iparser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/itt.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/network_metadata.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/npu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/parser_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/igraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu.cpp
)
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
)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})

ov_developer_package_export_targets(TARGET openvino::npu_common
                                    INSTALL_INCLUDE_DIRECTORIES
                                        ${CMAKE_CURRENT_SOURCE_DIR}/include/)
