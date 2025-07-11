# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Use Clang for cross-compilation
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set the target triple for ARM64
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)

# Use all LLVM tools
set(CMAKE_AR llvm-ar)
set(CMAKE_RANLIB llvm-ranlib)
set(CMAKE_STRIP llvm-strip)
set(CMAKE_NM llvm-nm)
set(CMAKE_OBJDUMP llvm-objdump)
set(CMAKE_OBJCOPY llvm-objcopy)

# Use Clang's integrated assembler
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_ASM_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_ASM_FLAGS "-target aarch64-linux-gnu")
set(CMAKE_ASM_FLAGS_INIT "-target aarch64-linux-gnu")

# Force Clang to use integrated assembler
set(CMAKE_ASM_COMPILER_ARG1 "-target aarch64-linux-gnu -integrated-as")

# Use lld linker
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")

# Find programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Find libraries and headers in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Don't use GNU pkg-config for cross-compilation
# set(PKG_CONFIG_EXECUTABLE aarch64-linux-gnu-pkg-config CACHE PATH "Path to ARM64 pkg-config")

# Disable components that have issues with cross-compilation
# MLAS uses assembly syntax that is incompatible with LLVM assembler for ARM64
set(ENABLE_MLAS_FOR_CPU OFF CACHE BOOL "Disable MLAS due to assembly syntax incompatibility with LLVM" FORCE)

# KleidiAI requires specific ARM assembly instructions that fail with LLVM cross-compilation
set(ENABLE_KLEIDIAI_FOR_CPU OFF CACHE BOOL "Disable KleidiAI due to assembly syntax issues" FORCE)

# TensorFlow Lite frontend builds flatc compiler for target architecture instead of host
# This causes "Exec format error" when trying to run ARM64 flatc on x86_64 host during build
set(ENABLE_OV_TF_LITE_FRONTEND OFF CACHE BOOL "Disable TF Lite frontend due to flatc cross-compilation issues" FORCE)
