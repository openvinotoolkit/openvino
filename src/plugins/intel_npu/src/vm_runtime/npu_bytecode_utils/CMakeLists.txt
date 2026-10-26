#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME npu_bytecode_utils)

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

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::runtime
)

target_include_directories(${TARGET_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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
