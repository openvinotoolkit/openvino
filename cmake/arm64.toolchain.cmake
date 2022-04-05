# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(PKG_CONFIG_EXECUTABLE aarch64-linux-gnu-pkg-config CACHE PATH "Path to ARM64 pkg-config")

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
    elseif(CMAKE_HOST_APPLE)
        SET(APPLE 1)
        SET(UNIX)
    endif()
    find_program(${ARGN})
    SET(WIN32)
    SET(APPLE)
    SET(UNIX 1)
    __cmake_find_root_restore()
endmacro()

# macro to find packages on the host OS
macro(find_host_package)
    __cmake_find_root_save_and_reset()
    if(CMAKE_HOST_WIN32)
        SET(WIN32 1)
        SET(UNIX)
    elseif(CMAKE_HOST_APPLE)
        SET(APPLE 1)
        SET(UNIX)
    endif()
    find_package(${ARGN})
    SET(WIN32)
    SET(APPLE)
    SET(UNIX 1)
    __cmake_find_root_restore()
endmacro()
