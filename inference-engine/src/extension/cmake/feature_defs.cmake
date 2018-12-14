# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8)

macro(ext_message TRACE_LEVEL)
    if (${TRACE_LEVEL} STREQUAL FATAL_ERROR)
        if(InferenceEngine_FIND_REQUIRED)
            message(FATAL_ERROR "${ARGN}")
        elseif(NOT InferenceEngine_FIND_QUIETLY)
            message(WARNING "${ARGN}")
        endif()
        return()
    elseif(NOT InferenceEngine_FIND_QUIETLY)
        message(${TRACE_LEVEL} "${ARGN}")
    endif ()
endmacro()

include(CPUID)
include(OptimizationFlags)

macro(enable_omp)
    if(UNIX) # Linux
        add_definitions(-fopenmp)
        find_library(intel_omp_lib iomp5
            PATHS ${InferenceEngine_INCLUDE_DIRS}/../external/omp/lib
        )
    elseif(WIN32) # Windows
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
            set(OPENMP_FLAGS "/Qopenmp /openmp")
            set(CMAKE_SHARED_LINKER_FLAGS " ${CMAKE_SHARED_LINKER_FLAGS} /nodefaultlib:vcomp")
        elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
            set(OPENMP_FLAGS "/Qopenmp /openmp")
        else()
            ext_message(WARNING "Unknown compiler ID. OpenMP support is disabled.")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")
        find_library(intel_omp_lib
            libiomp5md
            PATHS "${InferenceEngine_INCLUDE_DIRS}/../lib/intel64/${CMAKE_BUILD_TYPE}"
        )
    endif()
endmacro(enable_omp)
