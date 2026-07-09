# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_compiler_adapter")

set(SOURCES
    include/compiler_impl.hpp
    include/custom_stream_buffer.hpp
    include/driver_compiler_adapter.hpp
    include/dynamic_graph.hpp
    include/graph.hpp
    include/mem_usage.hpp
    include/model_serializer.hpp
    include/parser.hpp
    include/plugin_compiler_adapter.hpp
    include/weightless_graph.hpp
    include/weightless_utils.hpp
    include/xml_serializer.hpp
    include/ze_graph_ext_wrappers.hpp
    src/compiler_adapter_factory.cpp
    src/compiler_impl.cpp
    src/driver_compiler_adapter.cpp
    src/dynamic_graph.cpp
    src/graph.cpp
    src/mem_usage.cpp
    src/model_serializer.cpp
    src/parser.cpp
    src/parser_factory.cpp
    src/plugin_compiler_adapter.cpp
    src/precomp.hpp
    src/weightless_graph.cpp
    src/weightless_utils.cpp
    src/xml_serializer.cpp
    src/ze_graph_ext_wrappers.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES} $<TARGET_OBJECTS:openvino_npu_vcl_utils>)

ov_build_target_faster(${TARGET_NAME}
    PCH_HEADER "src/precomp.hpp"
)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        $<TARGET_PROPERTY:openvino_npu_vm_utils,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino_npu_common
        openvino::npu_al
        openvino::xml_util
)

#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})
