# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_npu_vm_utils)

file(GLOB_RECURSE SOURCES *.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})

set_target_properties(
  ${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

set_property(TARGET openvino_npu_vm_utils APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:LevelZero::LevelZero,INTERFACE_INCLUDE_DIRECTORIES>)
add_library(openvino::npu_vm_utils ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${NPU_UTILS_SOURCE_DIR}/include>
        s$<TARGET_PROPERTY:ze_loader,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(${TARGET_NAME} PUBLIC 
openvino::util 
openvino::core::dev 
openvino::runtime::dev
npu_vm_runtime_api
openvino_npu_al # TODO: need remove #include "intel_npu/npu_mlir_runtime.hpp"?
openvino_npu_zero_utils
)

#
# targets install
#

ov_install_static_lib(${TARGET_NAME} ${NPU_PLUGIN_COMPONENT})