# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The two functions below are used to allow cmake find search for host system
# locations during find_package

macro(ov_cross_compile_define_debian_arch)
    if(CMAKE_HOST_LINUX AND CMAKE_CROSSCOMPILING)
        set(_old_CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_LIBRARY_ARCHITECTURE})
        # without this WA cmake does not search in <triplet> subfolder
        # see https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
        if(HOST_X86_64)
            set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
        elseif(HOST_AARCH64)
            set(CMAKE_LIBRARY_ARCHITECTURE "aarch64-linux-gnu")
        elseif(HOST_RISCV64)
            set(CMAKE_LIBRARY_ARCHITECTURE "riscv64-linux-gnu")
        endif()
    endif()
endmacro()

macro(ov_cross_compile_define_debian_arch_reset)
    if(CMAKE_HOST_LINUX AND CMAKE_CROSSCOMPILING)
        set(CMAKE_LIBRARY_ARCHITECTURE ${_old_CMAKE_LIBRARY_ARCHITECTURE})
        unset(_old_CMAKE_LIBRARY_ARCHITECTURE)
    endif()
endmacro()

# Search packages for the host system instead of packages for the target system
# in case of cross compilation these macros should be defined by the toolchain file

if(CMAKE_CROSSCOMPILING AND NOT (OV_ARCH STREQUAL OV_HOST_ARCH AND
                                 CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME))
    # don't look at directories which are part of PATH (with removed bin / sbin at the end)
    # like /opt/homebrew on macOS where we cannot use system env path, because brew's
    # dependencies will be found, but at the same time we need to find flatbufffers and
    # other build system dependencies
    # ov_set_if_not_defined(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
    ov_set_if_not_defined(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)
    # it contains /usr and if we set this var to OFF, then CMAKE_FIND_ROOT_PATH is ignored
    # ov_set_if_not_defined(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)
    if(LINUX)
        # set root paths (overridden to /usr/lib/<CMAKE_LIBRARY_ARCHITECTURE>/cmake)
        # CMAKE_LIBRARY_ARCHITECTURE is defined automatically by cmake after trying the compilers
        # ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH "/usr")
    endif()

    # controling CMAKE_FIND_ROOT_PATH usage
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

macro(__ov_cmake_find_system_path_save_and_reset)
    foreach(v
            CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH
            CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY
            CMAKE_FIND_USE_CMAKE_SYSTEM_PATH
            )
        if(DEFINED ${v})
            set(__ov_save_${v} ${${v}})
        else()
            set(__ov_save_${v} ON)
        endif()
        set(${v} ON)
    endforeach()
endmacro()

macro(__ov_cmake_find_system_path_restore)
    foreach(v
            CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH
            CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY
            CMAKE_FIND_USE_CMAKE_SYSTEM_PATH
            )
        set(${v} ${__ov_save_${v}})
        unset(__ov_save_${v})
    endforeach()
endmacro()

macro(__ov_cmake_find_root_save_and_reset)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(__ov_save_${v} ${${v}})
        set(${v} NEVER)
    endforeach()
endmacro()

macro(__ov_cmake_find_root_restore)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(${v} ${__ov_save_${v}})
        unset(__ov_save_${v})
    endforeach()
endmacro()

macro(__ov_cmake_target_flags_save_and_reset)
    foreach(v WIN32 UNIX LINUX APPLE ANDROID BSD)
        set(__ov_target_save_${v} ${${v}})
        unset(${v})
    endforeach()

    if(CMAKE_HOST_WIN32)
        set(WIN32 1)
    elseif(CMAKE_HOST_APPLE)
        set(APPLE 1)
        set(UNIX 1)
    elseif(CMAKE_HOST_LINUX)
        set(LINUX 1)
        set(UNIX 1)
    elseif(CMAKE_HOST_UNIX)
        set(UNIX 1)
    elseif(CMAKE_HOST_BSD)
        set(BSD 1)
    endif()
endmacro()

macro(__ov_cmake_target_flags_restore)
    foreach(v WIN32 UNIX LINUX APPLE ANDROID BSD)
        set(${v} ${__ov_target_save_${v}})
        unset(__ov_target_save_${v})
    endforeach()
endmacro()

if(CMAKE_CROSSCOMPILING)
    # macro to find programs on the host OS
    if(NOT COMMAND find_host_package)
        macro(find_host_package)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_package(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
    if(NOT COMMAND find_host_program)
        macro(find_host_program)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_program(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
    if(NOT COMMAND find_host_library)
        macro(find_host_library)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_library(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
else()
    if(NOT COMMAND find_host_package)
        macro(find_host_package)
            find_package(${ARGN})
        endmacro()
    endif()
    if(NOT COMMAND find_host_program)
        macro(find_host_program)
            find_program(${ARGN})
        endmacro()
    endif()
    if(NOT COMMAND find_host_library)
        macro(find_host_library)
            find_library(${ARGN})
        endmacro()
    endif()
endif()
