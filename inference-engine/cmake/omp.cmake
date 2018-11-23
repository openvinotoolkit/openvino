# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

if (APPLE OR WIN32)

    find_path(OMP_INC omp.h)
    find_library(OMP_LIB iomp5
        PATHS   ${OMP}/lib)

    if (OMP_INC AND OMP_LIB)
        set(HAVE_OMP TRUE)
        get_filename_component(OMP_LIB_DIR "${OMP_LIB}" PATH)
    else()
        if (THREADING STREQUAL "OMP")
            find_package(OpenMP)
            if (NOT OPENMP_FOUND)    
                message(WARNING "OpenMP not found. OpenMP support will be disabled.")
            endif()
        endif()
    endif()
endif()


macro(enable_omp)
    if (APPLE) ## MacOS
        if (HAVE_OMP)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libiomp5")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${OMP_LIB_DIR}")
        else()
            message(WARNING "Was trying to enable OMP for some target. However OpenMP was not detected on system.")
        endif()
    elseif(UNIX) # Linux
        add_definitions(-fopenmp)
    elseif(WIN32) # Windows
        if (THREADING STREQUAL "OMP")
            set(OPENMP_FLAGS "/Qopenmp /openmp")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS} ${OPENMP_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS} ${OPENMP_FLAGS}")
        endif()
    endif()

    if (ENABLE_INTEL_OMP)
        if (WIN32)
            find_library(intel_omp_lib
                libiomp5md
                PATHS ${OMP}/lib ${ICCLIB})
            set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /nodefaultlib:vcomp")
            set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /nodefaultlib:vcomp")
        else()
            find_library(intel_omp_lib
                    iomp5
                    PATHS ${OMP}/lib)
        endif()
    endif()
endmacro(enable_omp)
