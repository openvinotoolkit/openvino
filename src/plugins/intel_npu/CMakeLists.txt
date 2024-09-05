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
set(NPU_INTERNAL_COMPONENT "${NPU_PLUGIN_COMPONENT}_internal")

set(NPU_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/features.cmake)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(CMAKE_COMPILE_WARNING_AS_ERROR AND WIN32)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        if(CMAKE_VERSION VERSION_LESS 3.24)
            ov_add_compiler_flags(/WX)
        endif()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WX")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /WX")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        ov_add_compiler_flags(-Werror)
    endif()
endif()

if(ENABLE_NPU_DEBUG_CAPS)
    add_compile_definitions(NPU_PLUGIN_DEVELOPER_BUILD)
endif()

add_subdirectory(thirdparty EXCLUDE_FROM_ALL)

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_INTEL_NPU_INTERNAL)
    add_subdirectory(tools)

    ov_cpack_add_component(${NPU_INTERNAL_COMPONENT} HIDDEN)
endif()
