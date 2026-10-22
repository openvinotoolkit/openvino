# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_npu_vm_runtime")

set(SOURCES
    src/npu_vm_runtime.cpp
    src/virtual_machine.cpp
    src/utils/buffer.cpp
    src/utils/buffer_metadata.cpp
    src/utils/call_frame.cpp
    src/utils/function.cpp
    src/utils/host_allocator.cpp
    src/utils/kernel_execution.cpp
    src/utils/level_zero_allocator.cpp
    src/utils/math.cpp
    src/utils/parameters.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} SHARED ${SOURCES})
add_library(openvino::intel_npu_vm_runtime ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME intel_npu_vm_runtime)
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# To fix BinSkim scan issues in the CID package test nightly job
# Limit LTCG to this library to avoid changing linker flags for other targets in the directory scope.
if(MSVC)
    target_link_options(${TARGET_NAME}
        PRIVATE
            $<$<CONFIG:Release>:/LTCG>
    )
endif()

target_link_libraries(${TARGET_NAME}
    PUBLIC
        openvino::intel_npu_bytecode_utils
    PRIVATE
        openvino::zero_loader
        openvino::npu_zero_utils
        openvino::runtime
        openvino::util
        openvino::npu_al
)

# Prevents warning about unused parameters in NPU plugin logger header:
#   include/openvino/core/attribute_adapter.hpp:38:44: error: unused parameter 'x' [-Werror=unused-parameter]
if(NOT MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wno-error=unused-parameter
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT ENABLE_DEVELOPER_BUILD)
    target_compile_options(${TARGET_NAME} PRIVATE -DBUILD_LOG_LEVEL=4)
else()
    target_compile_options(${TARGET_NAME} PRIVATE -DBUILD_LOG_LEVEL=6)
endif()

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
    INSTALL_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
