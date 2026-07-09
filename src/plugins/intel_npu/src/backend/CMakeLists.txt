# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_npu_level_zero_backend")

set(SOURCES
    include/zero_backend.hpp
    include/zero_device.hpp
    include/zero_dynamic_infer_request.hpp
    include/zero_dynamic_pipeline.hpp
    include/zero_infer_request.hpp
    include/zero_pipeline.hpp
    include/zero_profiling.hpp
    include/zero_variable_state.hpp
    src/zero_backend.cpp
    src/zero_device.cpp
    src/zero_dynamic_infer_request.cpp
    src/zero_dynamic_pipeline.cpp
    src/zero_infer_request.cpp
    src/zero_pipeline.cpp
    src/zero_profiling.cpp
    src/zero_variable_state.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES} $<TARGET_OBJECTS:openvino_npu_vm_utils>)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)

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
)
#
# targets install
#
ov_install_static_lib(${TARGET_NAME} ${NPU_INTERNAL_COMPONENT})
