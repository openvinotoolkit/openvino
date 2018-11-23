# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
include("features")
include("mode")
if (THREADING STREQUAL "OMP")
    include("omp")
endif()
include("itt")

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

if (ARCH_64)
else()
    if (UNIX OR APPLE)
        SET(ENABLE_CLDNN OFF)
    endif()
    SET(ENABLE_MKL_DNN OFF)
endif()


#apple specific
if (APPLE)
    set(ENABLE_CLDNN OFF)
endif()


#minGW specific - under wine no support for downloading file and applying them using git
if (WIN32)
    enable_omp()

    if (MINGW)
        SET(ENABLE_CLDNN OFF) # dont have mingw dll for linking
        set(ENABLE_SAMPLES OFF)
    endif()
endif()

# Linux specific - not all OS'es are supported
if (LINUX)
    include("linux_name")
    get_linux_name(LINUX_OS_NAME)
    if (LINUX_OS_NAME)
        if (NOT(
                ${LINUX_OS_NAME} STREQUAL "Ubuntu 14.04" OR
                ${LINUX_OS_NAME} STREQUAL "Ubuntu 16.04" OR
                ${LINUX_OS_NAME} STREQUAL "CentOS 7"))
        endif()
    else ()
        message(WARNING "Cannot detect Linux OS via reading /etc/*-release:\n ${release_data}")
    endif ()
endif ()

if (NOT ENABLE_MKL_DNN)
    set(GEMM OPENBLAS)
endif()

#next section set defines to be accesible in c++/c code for certain feature
if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_GTEST_PATCHES)
    add_definitions(-DENABLE_GTEST_PATCHES=1)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_MKL_DNN)
    add_definitions(-DENABLE_MKL_DNN=1)
endif()

if (ENABLE_STRESS_UNIT_TESTS)
    add_definitions(-DENABLE_STRESS_UNIT_TESTS=1)
endif()

if (ENABLE_SEGMENTATION_TESTS)
    add_definitions(-DENABLE_SEGMENTATION_TESTS=1)
endif()

if (ENABLE_OBJECT_DETECTION_TESTS)
    add_definitions(-DENABLE_OBJECT_DETECTION_TESTS=1)
endif()

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

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE  ON)
endif()

if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "SEQ")
    set(ENABLE_INTEL_OMP OFF)
    message(STATUS "ENABLE_INTEL_OMP should be disabled if THREADING is TBB or Sequential. ENABLE_INTEL_OMP option is " ${ENABLE_INTEL_OMP})
endif()

print_enabled_features()