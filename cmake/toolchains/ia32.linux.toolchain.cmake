# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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

# need libusb 32-bits version
_set_if_not_defined(ENABLE_INTEL_MYRIAD_COMMON OFF)
