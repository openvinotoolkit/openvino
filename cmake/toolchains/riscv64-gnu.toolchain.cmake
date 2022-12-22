# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: use with the following docker image https://github.com/Incarnation-p-lee/riscv-docker-emulator#llvm-clang-tool-chain

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV_TOOLCHAIN_ROOT "/opt/riscv/gnu-toolchain/rv64-linux" CACHE PATH "Path to CLANG for RISC-V cross compiler build directory")
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot" CACHE PATH "RISC-V sysroot")

set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_STRIP ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-strip)
set(CMAKE_AR ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc-ar)
set(CMAKE_RANLIB ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-ranlib)
set(CMAKE_ADDR2LINE ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-addr2line)
set(CMAKE_NM ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-nm)
set(CMAKE_LINKER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-ld)
set(CMAKE_OBJCOPY ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-objcopy)
set(CMAKE_OBJDUMP ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-objdump)
set(CMAKE_READELF ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-readelf)
set(PKG_CONFIG_EXECUTABLE "NOT-FOUND" CACHE PATH "Path to ARM64 pkg-config")

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_SHARED_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")

set(CMAKE_C_STANDARD_LIBRARIES_INIT "-latomic" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "-latomic" CACHE STRING "" FORCE)

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
