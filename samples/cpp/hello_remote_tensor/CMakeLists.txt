# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(OpenVINO_SOURCE_DIR AND NOT ENABLE_INTEL_GPU)
    message(STATUS "hello_remote_tensor SKIPPED: GPU plugin is not built")
    return()
endif()

# Only the Level Zero headers are needed: the loader is opened at runtime, because it ships with the GPU driver.
if(TARGET LevelZero::Headers)
    set(level_zero_headers LevelZero::Headers)
else()
    find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h)
    if(NOT LEVEL_ZERO_INCLUDE_DIR)
        message(STATUS "hello_remote_tensor SKIPPED: Level Zero headers are required")
        return()
    endif()

    add_library(level_zero_headers INTERFACE IMPORTED)
    set_target_properties(level_zero_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LEVEL_ZERO_INCLUDE_DIR}")
    set(level_zero_headers level_zero_headers)
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
              DEPENDENCIES format_reader ie_samples_utils ${level_zero_headers} OpenCL::OpenCL ${CMAKE_DL_LIBS})
