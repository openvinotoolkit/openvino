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

# Set both options explicitly: BUILD_STATIC_NO_DLL forces BUILD_STATIC into the cache,
# so a stale value would otherwise survive turning ENABLE_NPU_STATIC_ZE_LOADER off
if(ENABLE_NPU_STATIC_ZE_LOADER)
    set(BUILD_STATIC ON)
    set(BUILD_STATIC_NO_DLL ON)
    # Level Zero uses the static CRT by default, OpenVINO uses the dynamic one
    set(MSVC_BUILD_L0_DYNAMIC_VCRUNTIME ON)
else()
    set(BUILD_STATIC OFF)
    set(BUILD_STATIC_NO_DLL OFF)
endif()
add_subdirectory(level-zero EXCLUDE_FROM_ALL)

if(ENABLE_NPU_STATIC_ZE_LOADER)
    # ze_loader links level_zero_utils privately, so it has to be exportable too.
    # Its include directory points into the source tree, which install(EXPORT) rejects.
    get_target_property(_ze_utils_include_dirs level_zero_utils INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(level_zero_utils PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${_ze_utils_include_dirs}>")
    unset(_ze_utils_include_dirs)

    # Level Zero compiles with /GL for its own DLL; in a static library that would force /LTCG on the NPU plugin link,
    # which then fails on LNK4075 (/INCREMENTAL ignored) under /WX
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(ze_loader PRIVATE /GL-)
        target_compile_options(level_zero_utils PRIVATE /GL-)
    endif()

    # The Level Zero headers declare the API dllexport, which would export it from the NPU plugin DLL. The embedded
    # loader calls its functions directly rather than through GetProcAddress, so it doesn't need the exports
    target_compile_definitions(ze_loader PRIVATE ZE_APIEXPORT= ZE_DLLEXPORT=)
endif()

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

# Install Level Zero loader next to the GPU plugin on all supported platforms.
# A static loader is linked into the NPU plugin instead, so there is no DLL to install
# and the GPU plugin falls back to the ze_loader.dll installed with the driver.
if(NOT ENABLE_NPU_STATIC_ZE_LOADER)
    install(TARGETS ze_loader
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR}
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
        COMPONENT gpu)
endif()

# This VERSION file created by L0 may cause compilation issue of oneTBB headers, so remove it
file(REMOVE "${CMAKE_BINARY_DIR}/VERSION")
