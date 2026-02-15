# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Prerequisites:
#
# Build platform: Ubuntu
# apt-get install mingw-w64 mingw-w64-tools g++-mingw-w64-x86-64 gcc-mingw-w64-x86-64
#
# Build platform: macOS
# brew install mingw-w64
#

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(PKG_CONFIG_EXECUTABLE x86_64-w64-mingw32-pkg-config CACHE PATH "Path to Windows x86_64 pkg-config")
