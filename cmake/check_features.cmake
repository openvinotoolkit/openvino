# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE  ON)
endif()

# FIXME: there are compiler failures with LTO and Cross-Compile toolchains. Disabling for now, but
#        this must be addressed in a proper way
if(CMAKE_CROSSCOMPILING OR NOT (LINUX OR WIN32))
    set(ENABLE_LTO OFF)
endif()

#64 bits platform
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Detected 64 bit architecture")
    SET(ARCH_64 ON)
else()
    message(STATUS "Detected 32 bit architecture")
    SET(ARCH_64 OFF)
endif()

# 32 bits
if(NOT ARCH_64)
    if(UNIX)
        set(ENABLE_CLDNN OFF)
    endif()
    set(ENABLE_MKL_DNN OFF)
endif()

# Apple specific
if (APPLE)
    set(ENABLE_CLDNN OFF)
endif()

# ARM specific
if (ARM OR AARCH64)
    # disable all base plugins but Myriad
    set(ENABLE_CLDNN OFF)
    set(ENABLE_MKL_DNN OFF)
endif()

#minGW specific - under wine no support for downloading file and applying them using git
if (WIN32)
    if (MINGW)
        SET(ENABLE_CLDNN OFF) # dont have mingw dll for linking
    endif()
endif()

if (NOT ENABLE_MKL_DNN)
    set(ENABLE_MKL OFF)
endif()

print_enabled_features()
