# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# A status message is easy to miss, so fail when the sample was explicitly requested via BUILD_SAMPLE_NAME.
function(hello_remote_tensor_unavailable reason)
    if(BUILD_SAMPLE_NAME AND "hello_remote_tensor" IN_LIST BUILD_SAMPLE_NAME)
        message(FATAL_ERROR "hello_remote_tensor was requested via BUILD_SAMPLE_NAME but cannot be built: ${reason}")
    endif()
    message(STATUS "hello_remote_tensor SKIPPED: ${reason}")
endfunction()

# Level Zero headers and the loader library are build dependencies; the sample calls Level Zero directly.
set(LEVEL_ZERO_MIN_API_VERSION_MAJOR 1)
set(LEVEL_ZERO_MIN_API_VERSION_MINOR 11)

# The loader is always the system one (the Level Zero SDK or a distribution package), in-tree and standalone alike.
# It is provided by the GPU driver at run time, so no ze_loader library is built or copied next to the sample:
# a bundled loader could be older or newer than the driver and shadow the installed one.
# Standard CMake search (CMAKE_PREFIX_PATH, system paths) plus the location published by the Level Zero SDK
# installer in LEVEL_ZERO_V1_SDK_PATH. No hardcoded locations.
find_library(LEVEL_ZERO_LOADER_LIBRARY NAMES ze_loader
             HINTS ENV LEVEL_ZERO_V1_SDK_PATH
             PATH_SUFFIXES lib lib64)

if(TARGET LevelZero::Headers)
    # In-tree OpenVINO build, the same way as the GPU and NPU plugins consume Level Zero headers: they are staged
    # into the build tree by 'prepare_ze_headers' (after configure), so only the target carries their location.
    set(LEVEL_ZERO_HEADER_TARGET LevelZero::Headers)
    # ze_api.h to check the version in: the bundled submodule or the system package (pkg-config).
    find_file(LEVEL_ZERO_API_HEADER ze_api.h
              HINTS "${level_zero_INCLUDEDIR}"
                    "${OpenVINO_SOURCE_DIR}/thirdparty/level_zero/level-zero/include"
              PATH_SUFFIXES level_zero
              NO_DEFAULT_PATH)
else()
    find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h
              HINTS ENV LEVEL_ZERO_V1_SDK_PATH
              PATH_SUFFIXES include)
    if(LEVEL_ZERO_INCLUDE_DIR)
        set(LEVEL_ZERO_API_HEADER "${LEVEL_ZERO_INCLUDE_DIR}/level_zero/ze_api.h")
    endif()
endif()

if(NOT LEVEL_ZERO_API_HEADER)
    hello_remote_tensor_unavailable("Level Zero headers are required")
    return()
endif()
if(NOT LEVEL_ZERO_LOADER_LIBRARY)
    hello_remote_tensor_unavailable("the system Level Zero loader library (ze_loader) is required")
    return()
endif()

# Always verify the API version of the headers, whatever their source is.
file(STRINGS "${LEVEL_ZERO_API_HEADER}" _ze_version_line
     REGEX "#define[ \t]+ZE_API_VERSION_CURRENT_M[ \t]" LIMIT_COUNT 1)
string(REGEX MATCH "ZE_MAKE_VERSION\\([ \t]*([0-9]+)[ \t]*,[ \t]*([0-9]+)[ \t]*\\)" _ze_version "${_ze_version_line}")
if(NOT _ze_version)
    hello_remote_tensor_unavailable("cannot read Level Zero API version from ${LEVEL_ZERO_API_HEADER}")
    return()
endif()
set(_ze_major ${CMAKE_MATCH_1})
set(_ze_minor ${CMAKE_MATCH_2})
if(_ze_major LESS LEVEL_ZERO_MIN_API_VERSION_MAJOR OR
   (_ze_major EQUAL LEVEL_ZERO_MIN_API_VERSION_MAJOR AND _ze_minor LESS LEVEL_ZERO_MIN_API_VERSION_MINOR))
    hello_remote_tensor_unavailable("Level Zero API v${LEVEL_ZERO_MIN_API_VERSION_MAJOR}.${LEVEL_ZERO_MIN_API_VERSION_MINOR} "
                                    "or newer is required, found v${_ze_major}.${_ze_minor} in ${LEVEL_ZERO_API_HEADER}")
    return()
endif()
message(STATUS "hello_remote_tensor: Level Zero API v${_ze_major}.${_ze_minor} (${LEVEL_ZERO_API_HEADER})")

# OpenCL is a build dependency used for a single call, clGetContextInfo(CL_L0_CONTEXT_HANDLE), which converts the
# GPU plugin's OpenCL context to its Level Zero context on drivers with Level Zero - OpenCL interoperability (LEO).
# The cl_context comes from the GPU plugin, which links the same OpenCL::OpenCL in-tree, so the query goes through
# the same ICD loader that created the context.
if(NOT TARGET OpenCL::OpenCL)
    find_package(OpenCL QUIET)
    if(NOT TARGET OpenCL::OpenCL)
        hello_remote_tensor_unavailable("OpenCL headers and library are required")
        return()
    endif()
endif()

ov_add_sample(NAME hello_remote_tensor
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              INCLUDE_DIRECTORIES ${LEVEL_ZERO_INCLUDE_DIR}
              DEPENDENCIES format_reader ie_samples_utils ${LEVEL_ZERO_HEADER_TARGET} ${LEVEL_ZERO_LOADER_LIBRARY}
                           OpenCL::OpenCL ${CMAKE_DL_LIBS})
