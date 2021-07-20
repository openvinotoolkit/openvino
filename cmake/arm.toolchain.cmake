# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

#  set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
#  set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
#    set(CMAKE_CXX_FLAGS           "-mthumb ${CMAKE_CXX_FLAGS}")
#    set(CMAKE_C_FLAGS             "-mthumb ${CMAKE_C_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,nocopyreloc")
#    set(ARM_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
#  set(CMAKE_SHARED_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
#  set(CMAKE_MODULE_LINKER_FLAGS "${ARM_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
#  set(CMAKE_EXE_LINKER_FLAGS    "${ARM_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")

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
