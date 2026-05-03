# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(OV_RPI_TOOLCHAIN_PREFIX "aarch64-none-linux-gnu" CACHE STRING "GNU target triplet prefix")
set(OV_RPI_SYSROOT "" CACHE PATH "Optional Raspberry Pi ARM64 sysroot")
set(OV_RPI_EXECUTABLE_SUFFIX ".exe")
set(OV_RPI_EXTRA_TOOLS as ld)
set(OV_RPI_MISSING_TOOLCHAIN_MESSAGE "Install Arm GNU Toolchain for Windows or set OV_RPI_TOOLCHAIN_PREFIX.")
set(OV_RPI_LIBRARY_DIR_SUFFIXES lib64 usr/lib64 lib/aarch64-linux-gnu usr/lib/aarch64-linux-gnu)
set(OV_RPI_PKG_CONFIG_DIR_SUFFIXES
    usr/lib64/pkgconfig
    usr/lib/aarch64-linux-gnu/pkgconfig
    usr/lib/pkgconfig
    usr/share/pkgconfig)

include("${CMAKE_CURRENT_LIST_DIR}/rpi_cross/rpi-aarch64-linux-common.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rpi_cross/rpi-aarch64-linux-windows-wrappers.cmake")
