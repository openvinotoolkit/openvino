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

if (NOT ENABLE_MKL_DNN)
    set(ENABLE_MKL OFF)
endif()

if(ENABLE_AVX512F)
    if ((CMAKE_CXX_COMPILER_ID MATCHES MSVC) AND (MSVC_VERSION VERSION_LESS 1920))
        # 1920 version of MSVC 2019. In MSVC 2017 AVX512F not work
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
    if ((CMAKE_CXX_COMPILER_ID STREQUAL GNU) AND (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)))
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
endif()

print_enabled_features()
