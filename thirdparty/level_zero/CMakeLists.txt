# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(BUILD_SHARED_LIBS OFF)


# We have to avoid linking against loader with debug postfix due it's a part of driver
# and the name will be the same for release and debug configurations
set(CMAKE_DEBUG_POSTFIX "")

# Skip warnings as errors for thirdparty
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/WX-)
    # Close spectre for ze loader
    add_compile_options("/Qspectre-")

    string(REGEX REPLACE "[-/]INCREMENTAL" "/INCREMENTAL:NO" CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG}")
    string(REGEX REPLACE "[-/]INCREMENTAL" "/INCREMENTAL:NO" CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    string(REGEX REPLACE "[-/]INCREMENTAL" "/INCREMENTAL:NO" CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-error)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -Wno-undef \
        -Wno-missing-declarations")
    if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
    endif()
endif()
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
add_subdirectory(level-zero EXCLUDE_FROM_ALL)

set(ZE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/")
file(GLOB_RECURSE COMPUTE_RUNTIME_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/compute-runtime/*.h")
file(GLOB_RECURSE LEVEL_ZERO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include")
add_custom_command(OUTPUT "${ZE_INCLUDE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include" "${ZE_INCLUDE_DIR}/level_zero"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/compute-runtime" "${ZE_INCLUDE_DIR}/level_zero"
    DEPENDS "${COMPUTE_RUNTIME_HEADERS}" "${LEVEL_ZERO_HEADERS}"
    COMMENT "Copying Level Zero and compute-runtime headers..."
)
add_custom_target(prepare_ze_headers ALL DEPENDS "${ZE_INCLUDE_DIR}")
add_dependencies(ze_loader prepare_ze_headers)

# Allow include patterns with and without level-zero/ prefix
set_property(TARGET ze_loader APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${ZE_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ZE_INCLUDE_DIR}/level_zero>
)


# This VERSION file created by L0 may cause compilation issue of oneTBB headers, so remove it
file(REMOVE "${CMAKE_BINARY_DIR}/VERSION")
