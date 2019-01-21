# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

#module to locate GNA libraries

cmake_minimum_required(VERSION 2.8)

if (WIN32)
    set(GNA_PLATFORM_DIR win64)
    set(GNA_LIB_DIR x64)
    set(GNA_LIB gna)
elseif (UNIX)
    set(GNA_PLATFORM_DIR linux)
    set(GNA_LIB_DIR lib)
    set(GNA_LIB gna_api)
    set(GNA_KERNEL_LIB gna_kernel)
else ()
    message(FATAL_ERROR "GNA not supported on this platform, only linux, and windows")
endif ()

find_library(GNA_API_LIBRARY
        ${GNA_LIB}
        HINTS
        ${GNA}/${GNA_PLATFORM_DIR}/${GNA_LIB_DIR})

set(libGNA_INCLUDE_DIRS ${GNA}/${GNA_PLATFORM_DIR}/include)
set(libGNA_LIBRARY ${GNA_API_LIBRARY})

if (UNIX)
    #message("Searching for libgna_kernel.so in: ${GNA}/${GNA_PLATFORM_DIR}/${GNA_KERNEL_LIB}")
    find_library(GNA_KERNEL_LIBRARY
            ${GNA_KERNEL_LIB}
            HINTS
            ${GNA}/${GNA_PLATFORM_DIR}/${GNA_LIB_DIR})
endif ()

set(libGNA_LIBRARIES ${libGNA_LIBRARY} ${GNA_KERNEL_LIBRARY})
