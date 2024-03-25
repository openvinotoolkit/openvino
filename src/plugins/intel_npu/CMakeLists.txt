# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_NPU)
    return()
endif()

#
# Build properties
#

set(NPU_DEVICE_NAME "NPU")
string(TOLOWER "${NPU_DEVICE_NAME}" NPU_PLUGIN_COMPONENT)
set(NPU_PLUGIN_COMPONENT "${NPU_PLUGIN_COMPONENT}_internal")

set(NPU_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/features.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_DEVELOPER_BUILD)
    add_compile_definitions(NPU_PLUGIN_DEVELOPER_BUILD)
endif()

add_subdirectory(thirdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)
