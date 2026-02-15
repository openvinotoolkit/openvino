# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME WindowsStore)

#
# Define CMAKE_SYSTEM_VERSION if not defined
#

if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    # Sometimes CMAKE_HOST_SYSTEM_VERSION has form 10.x.y while we need
    # form 10.x.y.z Adding .0 at the end fixes the issue
    if(CMAKE_HOST_SYSTEM_VERSION MATCHES "^10\.0\.[0-9]+$")
        set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}.0")
    else()
        set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}")
    endif()
endif()

if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

#
# Compilation flags
#

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp"
    "#ifdef WINAPI_FAMILY\n"
    "#undef WINAPI_FAMILY\n"
    "#define WINAPI_FAMILY WINAPI_FAMILY_DESKTOP_APP\n"
    "#endif\n")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FI\"${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp\"")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /FI\"${CMAKE_CURRENT_BINARY_DIR}/src/uwp.hpp\"")

set(CMAKE_VS_GLOBALS "WindowsTargetPlatformMinVersion=${CMAKE_SYSTEM_VERSION}")
