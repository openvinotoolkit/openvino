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

if(ENABLE_NPU_DEBUG_CAPS)
    add_compile_definitions(NPU_PLUGIN_DEVELOPER_BUILD)
endif()

add_subdirectory(thirdparty EXCLUDE_FROM_ALL)

#
# Warnings as errors
#

macro(remove_compiler_flags FLAGS)
    string(REPLACE " " ";" _tokened_list "${FLAGS}")
    foreach(flag IN LISTS _tokened_list)
        string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE "${flag}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "${flag}" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    endforeach()
endmacro()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
    remove_compiler_flags(/wd4996)
    if(CMAKE_VERSION VERSION_LESS 3.24)
        ov_add_compiler_flags(/WX)
    endif()

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WX")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /WX")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
endif()

if(CMAKE_COMPILE_WARNING_AS_ERROR)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        remove_compiler_flags(-Wno-error=deprecated-declarations)
        ov_add_compiler_flags(-Werror)

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            remove_compiler_flags(/Qdiag-disable:1478,1786)
        else()
            remove_compiler_flags(-diag-disable=1478,1786)
        endif()
    endif()
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_INTEL_NPU_INTERNAL)
    add_subdirectory(tools)

    ov_cpack_add_component(${NPU_INTERNAL_COMPONENT} HIDDEN)
endif()
