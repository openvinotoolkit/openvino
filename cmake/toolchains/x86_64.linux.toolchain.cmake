# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Install compiler on debian using:
# apt-get install -y gcc-x86-64-linux-gnu g++-x86-64-linux-gnu binutils-x86-64-linux-gnu pkg-config-x86-64-linux-gnu

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR amd64)

set(CMAKE_C_COMPILER x86_64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER x86_64-linux-gnu-g++)
set(CMAKE_STRIP x86_64-linux-gnu-strip)
set(PKG_CONFIG_EXECUTABLE x86_64-linux-gnu-pkg-config CACHE PATH "Path to amd64 pkg-config")
