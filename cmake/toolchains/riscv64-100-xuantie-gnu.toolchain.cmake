# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# NOTE: use Xuantie compiler:
#    git clone https://github.com/XUANTIE-RV/xuantie-gnu-toolchain.git
#    ./configure --prefix=/opt/riscv
#    make linux
#    -DRISCV_TOOLCHAIN_ROOT=/opt/riscv

# To enable cross-compilation with python (for example, on Ubuntu 22.04):
# $ echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy main >> riscv64-sources.list
# $ echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy universe >> riscv64-sources.list
# $ echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main >> riscv64-sources.list
# $ echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> riscv64-sources.list
# $ mv riscv64-sources.list /etc/apt/sources.list.d/
# $ dpkg --add-architecture riscv64
# $ apt-get update -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/riscv64-sources.list
# $ apt-get install -y --no-install-recommends libpython3-dev:riscv64
# $ ln -s /usr/include/riscv64-linux-gnu/ /usr/include/python3.10/

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV64_XUANTIE ON)
set(RISCV64_RVV1p0 ON)

set(RISCV_TOOLCHAIN_ROOT $ENV{RISCV_TOOLCHAIN_ROOT} CACHE PATH "Path to GCC for RISC-V cross compiler build directory")
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot" CACHE PATH "RISC-V sysroot")

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_STRIP ${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-strip)
set(PKG_CONFIG_EXECUTABLE "NOT-FOUND" CACHE PATH "Path to RISC-V pkg-config")

set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -march=rv64gcv1p0_zfh -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -march=rv64gcv1p0_zfh -mabi=lp64d")

# system libc provides pthread functions (as detected by FindThreads.cmake), but not all functions are available
# WA: use pthread explicitly, since we know it's available in current toolchain
set(CMAKE_EXE_LINKER_FLAGS_INIT "-pthread")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-pthread")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-pthread")
