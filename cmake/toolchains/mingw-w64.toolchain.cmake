# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Prerequisites:
#
# Build platform: Ubuntu
# apt-get install mingw-w64 mingw-w64-tools g++-mingw-w64-x86-64 gcc-mingw-w64-x86-64
#
# Build platform: macOS
# brew install mingw-w64
#

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc-posix)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix)
set(PKG_CONFIG_EXECUTABLE x86_64-w64-mingw32-pkg-config CACHE PATH "Path to Windows x86_64 pkg-config")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

macro(__cmake_find_root_save_and_reset)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(__save_${v} ${${v}})
        set(${v} NEVER)
    endforeach()
endmacro()

macro(__cmake_find_root_restore)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(${v} ${__save_${v}})
        unset(__save_${v})
    endforeach()
endmacro()


# macro to find programs on the host OS
macro(find_host_program)
    __cmake_find_root_save_and_reset()
    if(CMAKE_HOST_WIN32)
        SET(WIN32 1)
        SET(UNIX)
        SET(APPLE)
    elseif(CMAKE_HOST_APPLE)
        SET(APPLE 1)
        SET(UNIX)
        SET(WIN32)
    elseif(CMAKE_HOST_UNIX)
        SET(UNIX 1)
        SET(WIN32)
        SET(APPLE)
    endif()
    find_program(${ARGN})
    SET(WIN32 1)
    SET(APPLE)
    SET(UNIX)
    __cmake_find_root_restore()
endmacro()

# macro to find packages on the host OS
macro(find_host_package)
    __cmake_find_root_save_and_reset()
    if(CMAKE_HOST_WIN32)
        SET(WIN32 1)
        SET(UNIX)
        SET(APPLE)
    elseif(CMAKE_HOST_APPLE)
        SET(APPLE 1)
        SET(WIN32)
        SET(UNIX)
    elseif(CMAKE_HOST_UNIX)
        SET(UNIX 1)
        SET(WIN32)
        SET(APPLE)
    endif()
    find_package(${ARGN})
    SET(WIN32 1)
    SET(APPLE)
    SET(UNIX)
    __cmake_find_root_restore()
endmacro()
