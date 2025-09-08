# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7l)

set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_STRIP arm-linux-gnueabihf-strip)
set(PKG_CONFIG_EXECUTABLE arm-linux-gnueabihf-pkg-config CACHE PATH "Path to ARM pkg-config")
