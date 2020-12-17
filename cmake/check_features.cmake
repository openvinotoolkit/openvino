# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "" FORCE)
endif()

#64 bits platform
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Detected 64 bit architecture")
    SET(ARCH_64 ON)
else()
    message(STATUS "Detected 32 bit architecture")
    SET(ARCH_64 OFF)
endif()

print_enabled_features()
