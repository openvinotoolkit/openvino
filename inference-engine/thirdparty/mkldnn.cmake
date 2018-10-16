# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET mkldnn)
set(MKLDNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mkl-dnn)

file(GLOB_RECURSE HDR
        ${MKLDNN_ROOT}/include/*.h
        ${MKLDNN_ROOT}/include/*.hpp
)
file(GLOB_RECURSE SRC
        ${MKLDNN_ROOT}/src/*.c
        ${MKLDNN_ROOT}/src/*.cpp
        ${MKLDNN_ROOT}/src/*.h
        ${MKLDNN_ROOT}/src/*.hpp
)
include_directories(
        ${MKLDNN_ROOT}/include
        ${MKLDNN_ROOT}/src
        ${MKLDNN_ROOT}/src/common
        ${MKLDNN_ROOT}/src/cpu/xbyak
)

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qlong-double")
    endif()
endif()

enable_omp()

add_library(${TARGET} STATIC ${HDR} ${SRC})

if(GEMM STREQUAL "OPENBLAS")
    ## enable cblas_gemm from OpenBLAS package
    add_definitions(-DUSE_CBLAS -D_SX)
    include_directories(${BLAS_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${BLAS_LIBRARIES})
else()
    ## enable cblas_gemm from mlkml package
    set(MKLROOT ${MKL})
    include(MKL.cmake)
endif()


target_link_libraries(${TARGET} ${${TARGET}_LINKER_LIBS})
