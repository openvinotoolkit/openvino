# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_intel_npu_bytecode_utils)

list(APPEND SOURCES
    src/bytecode_reader.cpp
    src/except.cpp
    src/format.cpp
    src/instructions.cpp
    src/logger.cpp
    src/magic_number.cpp
    src/network_description.cpp
    src/print_utils.cpp
    src/section_header_table.cpp
    src/type_section.cpp
    src/version.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC ${SOURCES})
add_library(openvino::intel_npu_bytecode_utils ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME intel_npu_bytecode_utils)

if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT ENABLE_DEVELOPER_BUILD)
    target_compile_options(${TARGET_NAME} PRIVATE -DBUILD_LOG_LEVEL=4)
else()
    target_compile_options(${TARGET_NAME} PRIVATE -DBUILD_LOG_LEVEL=6)
endif()

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::runtime
)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Prevents warning about unused parameters in NPU plugin logger header:
#   include/openvino/core/attribute_adapter.hpp:38:44: error: unused parameter 'x' [-Werror=unused-parameter]
if(NOT MSVC)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wno-error=unused-parameter
    )
endif()

# TODO
ov_developer_package_export_targets(TARGET ${TARGET_NAME}
    INSTALL_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/include>)
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
