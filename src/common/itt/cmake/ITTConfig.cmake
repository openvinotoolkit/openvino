# Copyright (C) 2018-2025 Intel Corporation
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
    elseif (EXISTS "$ENV{HOME}/intel/vtune_amplifier_2019")
        set(INTEL_VTUNE_DIR "$ENV{HOME}/intel/vtune_amplifier_2019")
    endif()
endif()

if(DEFINED INTEL_VTUNE_DIR)
    message(STATUS "INTEL_VTUNE_DIR = ${INTEL_VTUNE_DIR}")

    find_path(ITT_INCLUDE_DIR ittnotify.h
              PATHS "${INTEL_VTUNE_DIR}/include/")

    find_library(ITT_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}ittnotify${CMAKE_STATIC_LIBRARY_SUFFIX}"
                 PATHS "${INTEL_VTUNE_DIR}/lib64")

    set(Located_ITT_LIBS ${ITT_LIB})
    set(Located_ITT_INCLUDE_DIRS ${ITT_INCLUDE_DIR})
else()
    message(STATUS "INTEL_VTUNE_DIR is not defined")
endif()

# Handle find_package() arguments, and set ITT_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ITT
    REQUIRED_VARS
        Located_ITT_INCLUDE_DIRS
        Located_ITT_LIBS)

if(ITT_FOUND)
    set(INTEL_ITT_FOUND ${ITT_FOUND})

    add_library(ittapi::ittnotify STATIC IMPORTED GLOBAL)
    set_target_properties(ittapi::ittnotify PROPERTIES IMPORTED_LOCATION "${Located_ITT_LIBS}"
                                                       INTERFACE_INCLUDE_DIRECTORIES ${Located_ITT_INCLUDE_DIRS}
                                                       INTERFACE_COMPILE_DEFINITIONS ENABLE_PROFILING_ITT)

    if(UNIX)
        set_target_properties(ittapi::ittnotify PROPERTIES INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};Threads::Threads")
    endif()
endif()
