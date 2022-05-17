#===============================================================================
# Copyright 2017-2021 Intel Corporation
# Copyright 2021 FUJITSU LIMITED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage secure Development Lifecycle-related compiler flags
#===============================================================================

if(SDL_cmake_included)
    return()
endif()
set(SDL_cmake_included true)
include("cmake/utils.cmake")

# The flags that can be used for the main and host compilers should be moved to
# the macros to avoid code duplication and ensure consistency.
macro(sdl_unix_common_ccxx_flags var)
    append(${var} "-fPIC -Wformat -Wformat-security")
endmacro()

macro(sdl_gnu_common_ccxx_flags var)
    if (DNNL_DPCPP_HOST_COMPILER MATCHES "g\\+\\+")
        # GNU compiler 7.4 or newer is required for host compiler
        append(${var} "-fstack-protector-strong")
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            append(${var} "-fstack-protector-all")
        else()
            append(${var} "-fstack-protector-strong")
        endif()
    endif()
endmacro()

# GCC might be very paranoid for partial structure initialization, e.g.
#   struct { int a, b; } s = { 0, };
# However the behavior is triggered by `Wmissing-field-initializers`
# only. To prevent warnings on users' side who use the library and turn
# this warning on, let's use it too. Applicable for the library sources
# and interfaces only (tests currently rely on that fact heavily)
macro(sdl_gnu_src_ccxx_flags var)
    append(CMAKE_SRC_CCXX_FLAGS "-Wmissing-field-initializers")
endmacro()

macro(sdl_gnu_example_ccxx_flags var)
    # At this point the flags for src and examples are the same
    sdl_gnu_src_ccxx_flags(${var})
endmacro()

if(UNIX)
    set(CMAKE_CCXX_FLAGS)

    sdl_unix_common_ccxx_flags(CMAKE_CCXX_FLAGS)
    append(CMAKE_CXX_FLAGS_RELEASE "-D_FORTIFY_SOURCE=2")
    append(CMAKE_C_FLAGS_RELEASE "-D_FORTIFY_SOURCE=2")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        sdl_gnu_common_ccxx_flags(CMAKE_CCXX_FLAGS)
        sdl_gnu_src_ccxx_flags(CMAKE_SRC_CCXX_FLAGS)
        sdl_gnu_example_ccxx_flags(CMAKE_EXAMPLE_CCXX_FLAGS)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        get_filename_component(CXX_CMD_NAME ${CMAKE_CXX_COMPILER} NAME)
        # Fujitsu CXX compiler does not support "-fstack-protector-all".
        if(NOT CXX_CMD_NAME STREQUAL "FCC")
            append(CMAKE_CCXX_FLAGS "-fstack-protector-all")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        append(CMAKE_CXX_FLAGS "-fstack-protector")
    endif()
    append(CMAKE_C_FLAGS "${CMAKE_CCXX_FLAGS}")
    append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_FLAGS}")
    if(APPLE)
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-bind_at_load")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,-bind_at_load")
    else()
        append(CMAKE_EXE_LINKER_FLAGS "-pie")
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    endif()
elseif(MSVC AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CCXX_FLAGS "/guard:cf")
endif()
