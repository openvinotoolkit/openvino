# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if("${OV_ZE_INCLUDE_DIR}" STREQUAL "")
    set(ze_headers_source "${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include")
else()
    if(NOT EXISTS "${OV_ZE_INCLUDE_DIR}/level_zero/ze_api.h")
        message(FATAL_ERROR
            "Level Zero headers not found. Please check value of OV_ZE_INCLUDE_DIR."
            "(${OV_ZE_INCLUDE_DIR}/level_zero/ze_api.h does not exist)")
    endif()
    set(ze_headers_source "${OV_ZE_INCLUDE_DIR}/level_zero")
endif()

# Headers from the level zero submodule are directly inside "include" directory
# Copy them to "include/level_zero/" inside binary directory to allow level_zero/ prefix
set(ze_headers_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/include")
file(GLOB_RECURSE LEVEL_ZERO_HEADERS CONFIGURE_DEPENDS "${ze_headers_source}/*.h")
set(ZE_HEADERS_STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/ze_headers.stamp")
add_custom_command(OUTPUT "${ZE_HEADERS_STAMP_FILE}"
    DEPENDS ${LEVEL_ZERO_HEADERS}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${ze_headers_binary_dir}/level_zero"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${ze_headers_source}" "${ze_headers_binary_dir}/level_zero"
    COMMAND "${CMAKE_COMMAND}" -E touch "${ZE_HEADERS_STAMP_FILE}"
    COMMENT "Copying Level Zero headers..."
)
add_custom_target(prepare_ze_headers DEPENDS "${ZE_HEADERS_STAMP_FILE}")

add_library(level_zero_headers INTERFACE)
add_library(LevelZero::Headers ALIAS level_zero_headers)
add_dependencies(level_zero_headers prepare_ze_headers)

# OpenVINO includes Level Zero headers with the level_zero/ prefix
# thirdparty code includes them without the prefix. Expose both layouts.
target_include_directories(level_zero_headers INTERFACE
    $<BUILD_INTERFACE:${ze_headers_binary_dir}>
    $<BUILD_INTERFACE:${ze_headers_binary_dir}/level_zero>
)
ov_developer_package_export_targets(TARGET level_zero_headers
    INSTALL_INCLUDE_DIRECTORIES ${ze_headers_binary_dir})
ov_install_static_lib(level_zero_headers ${OV_CPACK_COMP_CORE})
