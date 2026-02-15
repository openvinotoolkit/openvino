# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: use with the following docker image https://github.com/Incarnation-p-lee/riscv-docker-emulator#gnu-toolchain

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV_TOOLCHAIN_ROOT "/opt/riscv/gnu-toolchain/rv64-linux" CACHE PATH "Path to GCC for RISC-V cross compiler build directory")
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
set(PKG_CONFIG_EXECUTABLE "NOT-FOUND" CACHE PATH "Path to RISC-V pkg-config")

set(CMAKE_SHARED_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-L${CMAKE_SYSROOT}/lib")

set(CMAKE_C_STANDARD_LIBRARIES_INIT "-latomic" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "-latomic" CACHE STRING "" FORCE)
