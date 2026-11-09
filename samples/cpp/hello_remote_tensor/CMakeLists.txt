# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(LEVEL_ZERO_INCLUDE_DIR "")

# Only the Level Zero headers are needed: the loader is opened at runtime, because it ships with the GPU driver.
if(TARGET LevelZero::Headers)
    get_target_property(LEVEL_ZERO_INCLUDE_DIRS LevelZero::Headers INTERFACE_INCLUDE_DIRECTORIES)
    foreach(_level_zero_dir IN LISTS LEVEL_ZERO_INCLUDE_DIRS)
        if(EXISTS "${_level_zero_dir}/level_zero/ze_api.h")
            set(LEVEL_ZERO_INCLUDE_DIR "${_level_zero_dir}")
            break()
        endif()
    endforeach()
endif()

if(NOT LEVEL_ZERO_INCLUDE_DIR)
    find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h PATH_SUFFIXES include)
endif()

if(NOT LEVEL_ZERO_INCLUDE_DIR OR NOT EXISTS "${LEVEL_ZERO_INCLUDE_DIR}/level_zero/ze_api.h")
    message(STATUS "hello_remote_tensor SKIPPED: Level Zero headers are required")
    return()
endif()

set(LEVEL_ZERO_MIN_API_VERSION_MAJOR 1)
set(LEVEL_ZERO_MIN_API_VERSION_MINOR 11)
set(LEVEL_ZERO_MIN_API_VERSION "${LEVEL_ZERO_MIN_API_VERSION_MAJOR}.${LEVEL_ZERO_MIN_API_VERSION_MINOR}")

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()
set(CMAKE_REQUIRED_INCLUDES "${LEVEL_ZERO_INCLUDE_DIR}")
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles("
    #include <level_zero/ze_api.h>
    #if ZE_API_VERSION_CURRENT_M < ZE_MAKE_VERSION(${LEVEL_ZERO_MIN_API_VERSION_MAJOR}, ${LEVEL_ZERO_MIN_API_VERSION_MINOR})
    #error \"Level Zero API v${LEVEL_ZERO_MIN_API_VERSION} or newer is required\"
    #endif
    int main() { return 0; }"
    LEVEL_ZERO_API_VERSION_OK)
cmake_pop_check_state()

if(NOT LEVEL_ZERO_API_VERSION_OK)
    message(STATUS "hello_remote_tensor SKIPPED: Level Zero API v${LEVEL_ZERO_MIN_API_VERSION} or newer is required")
    return()
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
