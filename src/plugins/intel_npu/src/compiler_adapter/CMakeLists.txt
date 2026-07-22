# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_compiler_adapter")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/compiler_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/custom_stream_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/driver_compiler_adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/dynamic_graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mem_usage.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/model_serializer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/parser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin_compiler_adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/weightless_graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/weightless_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/xml_serializer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ze_graph_ext_wrappers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_adapter_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compiler_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/driver_compiler_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dynamic_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mem_usage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_serializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parser_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin_compiler_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/precomp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/weightless_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/weightless_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xml_serializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ze_graph_ext_wrappers.cpp
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
