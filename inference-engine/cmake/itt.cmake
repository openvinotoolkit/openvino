# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

unset(ITT_INCLUDE_DIR CACHE)
unset(ITT_LIB CACHE)

if(NOT DEFINED INTEL_VTUNE_DIR AND DEFINED ENV{INTEL_VTUNE_DIR})
    set(INTEL_VTUNE_DIR "$ENV{INTEL_VTUNE_DIR}")
endif()
if(NOT DEFINED INTEL_VTUNE_DIR)
    if(EXISTS "/opt/intel/vtune_amplifier_xe/include")
    	set(INTEL_VTUNE_DIR "/opt/intel/vtune_amplifier_xe")
    elseif(EXISTS "/opt/intel/vtune_amplifier/include")
        set(INTEL_VTUNE_DIR "/opt/intel/vtune_amplifier")
    elseif (EXISTS "C:/Program Files (x86)/IntelSWTools/VTune Amplifier XE")
        set(INTEL_VTUNE_DIR "C:/Program Files (x86)/IntelSWTools/VTune Amplifier XE")
    elseif (EXISTS "C:/Program Files (x86)/IntelSWTools/VTune Amplifier")
        set(INTEL_VTUNE_DIR "C:/Program Files (x86)/IntelSWTools/VTune Amplifier")
    endif()
endif()

if(DEFINED INTEL_VTUNE_DIR)
    message(STATUS "INTEL_VTUNE_DIR = ${INTEL_VTUNE_DIR}")

    find_path(ITT_INCLUDE_DIR
    FILES
        ittnotify.h
    PATHS "${INTEL_VTUNE_DIR}/include/")

    find_library(ITT_LIB
    "libittnotify${CMAKE_STATIC_LIBRARY_SUFFIX}"
    PATHS ${INTEL_VTUNE_DIR}/lib64)

    set(Located_ITT_LIBS ${ITT_LIB} ${CMAKE_DL_LIBS})
    set(Located_ITT_INCLUDE_DIRS ${ITT_INCLUDE_DIR})
else()
    message(STATUS "INTEL_VTUNE_DIR is not defined")
endif()

# Handle find_package() arguments, and set INTEL_ITT_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(INTEL_ITT
    REQUIRED_VARS
        Located_ITT_INCLUDE_DIRS
        Located_ITT_LIBS)

if(ENABLE_PROFILING_ITT AND INTEL_ITT_FOUND)
    add_definitions(-DENABLE_PROFILING_ITT=1)

    set(INTEL_ITT_LIBS ${Located_ITT_LIBS})
    set(INTEL_ITT_INCLUDE_DIRS ${Located_ITT_INCLUDE_DIRS})

    message(STATUS "INTEL_ITT_INCLUDE_DIRS: ${INTEL_ITT_INCLUDE_DIRS}")
    include_directories(${INTEL_ITT_INCLUDE_DIRS})
    message(STATUS "INTEL_ITT_LIBS: ${INTEL_ITT_LIBS}")
else()
    add_definitions(-DENABLE_PROFILING_ITT=0)
    message(STATUS "INTEL_ITT is disabled")
endif()

