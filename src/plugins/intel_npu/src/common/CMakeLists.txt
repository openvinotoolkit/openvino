# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_common")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/blob_sections_registry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/blob_source.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/compiler_adapter_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/cre_token.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/cre.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/option_support_cache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/device_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/hsm_sections/encrypted_schedules_flag.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/icompiled_model.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/icompiler_adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/igraph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iprofiling_decoder.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/iparser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/isection_instance_evaluator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/isection_type_evaluator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/itt.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/major_minor_version.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/network_metadata.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/npu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/parser_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/hsm_sections/runtime_requirements.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/section_id.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/section_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_npu/common/single_section_instance_evaluator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blob_sections_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blob_source.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cre.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/encrypted_schedules_flag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/igraph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/isection_type_evaluator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/isection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/major_minor_version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network_metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_requirements.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/section_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/section_type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/single_section_instance_evaluator.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

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
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})
