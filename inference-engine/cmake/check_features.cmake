# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#64 bits platform
if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    message(STATUS "Detected 64 bit architecture")
    SET(ARCH_64 ON)
    SET(ARCH_32 OFF)
else()
    message(STATUS "Detected 32 bit architecture")
    SET(ARCH_64 OFF)
    SET(ARCH_32 ON)
endif()

if (NOT ARCH_64)
    if (UNIX OR APPLE)
        SET(ENABLE_CLDNN OFF)
    endif()
    SET(ENABLE_MKL_DNN OFF)
endif()

#apple specific
if (APPLE)
    set(ENABLE_GNA OFF)
    set(ENABLE_ROCKHOPER OFF)
    set(ENABLE_CLDNN OFF)
endif()


#minGW specific - under wine no support for downloading file and applying them using git
if (WIN32)
    if (MINGW)
        SET(ENABLE_CLDNN OFF) # dont have mingw dll for linking
        set(ENABLE_SAMPLES OFF)
    endif()
endif()

if (NOT ENABLE_MKL_DNN)
    set(ENABLE_MKL OFF)
endif()

if (NOT ENABLE_VPU)
    set(ENABLE_MYRIAD OFF)
endif()

#next section set defines to be accesible in c++/c code for certain feature
if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_MYRIAD)
    add_definitions(-DENABLE_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_MYRIAD )
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
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

    if (GNA_LIBRARY_VERSION STREQUAL "GNA2")
        message(WARNING "GNA2 is not currently supported. Fallback to ${DEFAULT_GNA_LIB}")
        set(GNA_LIBRARY_VERSION ${DEFAULT_GNA_LIB})
    endif()

    if (UNIX AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(WARNING "${GNA_LIBRARY_VERSION} no supported on GCC version ${CMAKE_CXX_COMPILER_VERSION}. Fallback to GNA1")
        set(GNA_LIBRARY_VERSION GNA1)
    endif()

    set(GNA_LIBRARY_VERSION "${GNA_LIBRARY_VERSION}" CACHE STRING "GNAVersion" FORCE)
    list (APPEND IE_OPTIONS GNA_LIBRARY_VERSION)
endif()

if (ENABLE_SAMPLES)
    set (ENABLE_SAMPLES_CORE ON)
endif()

#models dependend tests

if (DEVELOPMENT_PLUGIN_MODE)
    message (STATUS "Enabled development plugin mode")

    set (ENABLE_MKL_DNN OFF)
    set (ENABLE_TESTS OFF)

    message (STATUS "Initialising submodules")
    execute_process (COMMAND git submodule update --init ${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml
                     RESULT_VARIABLE git_res)

    if (NOT ${git_res})
        message (STATUS "Initialising submodules - done")
    endif()
endif()

if (NOT ENABLE_TESTS)
    set(ENABLE_GNA_MODELS OFF)
endif ()

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE  ON)
endif()


if(ENABLE_DUMP)
    add_definitions(-DDEBUG_DUMP)
endif()


print_enabled_features()
