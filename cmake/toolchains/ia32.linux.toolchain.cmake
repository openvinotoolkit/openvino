# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR i386)
set(CMAKE_STRIP i686-linux-gnu-strip)
set(PKG_CONFIG_EXECUTABLE i686-linux-gnu-pkg-config CACHE PATH "Path to 32-bits pkg-config")

set(CMAKE_CXX_FLAGS_INIT "-m32")
set(CMAKE_C_FLAGS_INIT "-m32")

set(CMAKE_SHARED_LINKER_FLAGS_INIT "-m32")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-m32")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-m32")

# Hints for OpenVINO

macro(_set_if_not_defined var val)
    if(NOT DEFINED ${var})
        set(${var} ${val} CACHE BOOL "" FORCE)
    endif()
endmacro()

# for ittapi
_set_if_not_defined(FORCE_32 ON)
