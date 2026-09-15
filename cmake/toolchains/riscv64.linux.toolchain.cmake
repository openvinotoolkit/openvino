# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Install compiler on debian using:
# apt-get install -y  gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER_TARGET riscv64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-linux-gnu)

set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_STRIP riscv64-linux-gnu-strip)

# GCC on RISC-V emits calls to __atomic_* libcalls for certain atomic ops.
# These are resolved by libatomic which must appear at the END of the link
# command (after static libraries that reference the symbols).
set(CMAKE_CXX_STANDARD_LIBRARIES "-latomic" CACHE STRING "" FORCE)
set(CMAKE_C_STANDARD_LIBRARIES "-latomic" CACHE STRING "" FORCE)
