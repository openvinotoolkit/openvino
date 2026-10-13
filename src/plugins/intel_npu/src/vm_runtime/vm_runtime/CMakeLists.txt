#
# Copyright (C) 2026 Intel Corporation
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
        npu_bytecode_utils
    PRIVATE
        openvino::ze_loader
        openvino::level-zero-ext
        openvino::npu_zero_utils
        openvino::runtime
        openvino::util
        npu_ov_utils
        openvino::npu_al
)

enable_warnings_as_errors(${TARGET_NAME} WIN_STRICT)
replace_noerror(${TARGET_NAME})

# Prevents warning about unused parameters in NPU plugin logger header:
#   include/openvino/core/attribute_adapter.hpp:38:44: error: unused parameter 'x' [-Werror=unused-parameter]
if(NOT MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wno-error=unused-parameter
    )
endif()

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<TARGET_PROPERTY:openvino::ze_loader,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::level-zero-ext,INTERFACE_INCLUDE_DIRECTORIES>
)

# include(${PROJECT_SOURCE_DIR}/cmake/npu_compiler_version.cmake)
npu_compiler_configure_version(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        LIBRARY DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${VPUX_INTERNAL_COMPONENT})

# Install openvino_intel_npu_vm_runtime target to the CID or CIP package selected by BUILD_COMPILER_FOR_DRIVER
install_target_with_debug_files(${TARGET_NAME} ${INSTALL_DESTINATION}/lib ${INSTALL_COMPONENT})

# Install openvino_intel_npu_vm_runtime target public header
install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/vm_runtime/npu_vm_runtime.hpp"
    DESTINATION ${INSTALL_DESTINATION}/include/vm_runtime
    COMPONENT ${INSTALL_COMPONENT})
