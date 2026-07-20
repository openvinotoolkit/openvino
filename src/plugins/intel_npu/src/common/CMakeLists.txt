# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_common")

set(SOURCES
    include/intel_npu/common/compiler_adapter_factory.hpp
    include/intel_npu/common/device_helpers.hpp
    include/intel_npu/common/filtered_config.hpp
    include/intel_npu/common/icompiled_model.hpp
    include/intel_npu/common/icompiler_adapter.hpp
    include/intel_npu/common/igraph.hpp
    include/intel_npu/common/iparser.hpp
    include/intel_npu/common/itt.hpp
    include/intel_npu/common/network_metadata.hpp
    include/intel_npu/common/npu.hpp
    include/intel_npu/common/parser_factory.hpp
    src/device_helpers.cpp
    src/filtered_config.cpp
    src/igraph.cpp
    src/network_metadata.cpp
    src/npu.cpp
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
