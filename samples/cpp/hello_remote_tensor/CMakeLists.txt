# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The sample works with any GPU runtime: it takes the plugin's default context and, when it is
# an OpenCL context, obtains the underlying Level Zero context through LEO.
if(OpenVINO_SOURCE_DIR AND NOT ENABLE_INTEL_GPU)
    message(STATUS "hello_remote_tensor SKIPPED: GPU plugin is not built")
    return()
endif()

if(NOT TARGET LevelZero::LevelZero)
    find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h)
    find_library(LEVEL_ZERO_LOADER NAMES ze_loader)
    if(NOT LEVEL_ZERO_INCLUDE_DIR OR NOT LEVEL_ZERO_LOADER)
        message(STATUS "hello_remote_tensor SKIPPED: Level Zero headers and loader are required")
        return()
    endif()

    add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
    set_target_properties(LevelZero::LevelZero PROPERTIES
        IMPORTED_LOCATION "${LEVEL_ZERO_LOADER}"
        INTERFACE_INCLUDE_DIRECTORIES "${LEVEL_ZERO_INCLUDE_DIR}")
endif()

# OpenCL is used only to get the Level Zero context on drivers with Level Zero - OpenCL interoperability (LEO)
if(NOT TARGET OpenCL::OpenCL)
    find_package(OpenCL QUIET)
    if(NOT TARGET OpenCL::OpenCL)
        message(STATUS "hello_remote_tensor SKIPPED: OpenCL headers and library are required")
        return()
    endif()
endif()

ov_add_sample(NAME hello_remote_tensor
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES format_reader ie_samples_utils LevelZero::LevelZero OpenCL::OpenCL)
