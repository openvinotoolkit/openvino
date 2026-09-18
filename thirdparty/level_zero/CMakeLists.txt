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
file(GLOB_RECURSE LEVEL_ZERO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include/*.h")
set(ZE_HEADERS_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/ze_headers.stamp)
add_custom_command(OUTPUT "${ZE_HEADERS_STAMP_FILE}"
    DEPENDS "${COMPUTE_RUNTIME_HEADERS}" "${LEVEL_ZERO_HEADERS}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/level-zero/include" "${ZE_INCLUDE_DIR}/level_zero"
    COMMAND "${CMAKE_COMMAND}" -E touch "${ZE_HEADERS_STAMP_FILE}"
    COMMENT "Copying Level Zero headers..."
)
add_custom_target(prepare_ze_headers ALL DEPENDS "${ZE_HEADERS_STAMP_FILE}")
add_dependencies(ze_loader prepare_ze_headers)

# OpenVINO prefers include pattern with level-zero pattern 
# include pattern without level_zero/ prefix is supported for thirdparty 
set_property(TARGET ze_loader APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    $<BUILD_INTERFACE:${ZE_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ZE_INCLUDE_DIR}/level_zero>
)

# The loader dlopens the tracing layer during zeInit and requires it to come from
# the same Level Zero release. Shipping the loader alone lets it pick up a
# different version from the system, which fails DDI table init with
# ZE_RESULT_ERROR_UNSUPPORTED_VERSION and makes zeInit report no GPU at all.
set(ze_installed_targets ze_loader)
if(TARGET ze_tracing_layer)
    set_target_properties(ze_tracing_layer PROPERTIES EXCLUDE_FROM_ALL OFF)
    add_dependencies(ze_loader ze_tracing_layer)
    list(APPEND ze_installed_targets ze_tracing_layer)

    # The loader reaches the tracing layer through dlopen(), which searches the
    # loader's own RUNPATH. Without it the sibling layer stays invisible and the
    # mismatched system-wide one is loaded instead.
    if(APPLE)
        set(ze_loader_rpath "@loader_path")
    elseif(NOT WIN32)
        set(ze_loader_rpath "$ORIGIN")
    endif()
    if(DEFINED ze_loader_rpath)
        set_property(TARGET ze_loader APPEND PROPERTY BUILD_RPATH "${ze_loader_rpath}")
        set_property(TARGET ze_loader APPEND PROPERTY INSTALL_RPATH "${ze_loader_rpath}")
    endif()
endif()

# Install Level Zero loader next to the GPU plugin on all supported platforms.
install(TARGETS ${ze_installed_targets}
    LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR}
    RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR}
    ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
    COMPONENT gpu)

# This VERSION file created by L0 may cause compilation issue of oneTBB headers, so remove it
file(REMOVE "${CMAKE_BINARY_DIR}/VERSION")
