# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Apple specific
if (APPLE)
    set(ENABLE_GNA OFF)
endif()


# Android specific
if(ANDROID)
    set(ENABLE_GNA OFF)
endif()

# ARM specific
if (ARM OR AARCH64)
    # disable all base plugins but Myriad
    set(ENABLE_GNA OFF)
    set(ENABLE_HDDL OFF)
endif()

# disable SSE
if(NOT(X86_64 OR X86))
    set(ENABLE_SSE42 OFF)
endif()

#minGW specific - under wine no support for downloading file and applying them using git
if (WIN32)
    if (MINGW)
        set(ENABLE_SAMPLES OFF)
    endif()
endif()

if (NOT ENABLE_VPU OR NOT ENABLE_NGRAPH)
    set(ENABLE_MYRIAD OFF)
endif()

if(CMAKE_CROSSCOMPILING)
    set(ENABLE_PROFILING_ITT OFF)
endif()

#next section set defines to be accesible in c++/c code for certain feature
if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_MYRIAD)
    add_definitions(-DENABLE_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_MYRIAD )
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
endif()

if (NOT ENABLE_TESTS)
    SET(ENABLE_BEH_TESTS OFF)
    SET(ENABLE_FUNCTIONAL_TESTS OFF)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_MKL_DNN)
    add_definitions(-DENABLE_MKL_DNN=1)
endif()

if (ENABLE_GNA)
    add_definitions(-DENABLE_GNA)

    set (DEFAULT_GNA_LIB GNA1_1401)

    # "GNA library version: GNA1|GNA1_1401|GNA2" - default is 1401
    if (NOT GNA_LIBRARY_VERSION STREQUAL "GNA1"
            AND NOT GNA_LIBRARY_VERSION STREQUAL "GNA1_1401"
            AND NOT GNA_LIBRARY_VERSION STREQUAL "GNA2")
        set (GNA_LIBRARY_VERSION ${DEFAULT_GNA_LIB})
        message(STATUS "GNA_LIBRARY_VERSION not set. Can be GNA1, GNA1_1401 or GNA2. Default is ${GNA_LIBRARY_VERSION}")
    endif()

    if (UNIX AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(WARNING "${GNA_LIBRARY_VERSION} no supported on GCC version ${CMAKE_CXX_COMPILER_VERSION}. Fallback to GNA1")
        set(GNA_LIBRARY_VERSION GNA1)
    endif()

    set(GNA_LIBRARY_VERSION "${GNA_LIBRARY_VERSION}" CACHE STRING "GNAVersion" FORCE)
    list (APPEND IE_OPTIONS GNA_LIBRARY_VERSION)
endif()

if(ENABLE_DUMP)
    add_definitions(-DDEBUG_DUMP)
endif()

if (LINUX AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
    set(ENABLE_UNICODE_PATH_SUPPORT OFF)
endif()

if (ENABLE_UNICODE_PATH_SUPPORT)
    add_definitions(-DENABLE_UNICODE_PATH_SUPPORT=1)
endif()

# functional tests require FormarParser which is disabled by this option
if(NOT ENABLE_IR_READER)
    set(ENABLE_FUNCTIONAL_TESTS OFF)
endif()

print_enabled_features()
