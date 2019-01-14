# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#  Brief description: This cmake file replase original mkl-dnn build scripts
#  for more convenient integration to IE build process
#
#===============================================================================

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET mkldnn)
set(MKLDNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mkl-dnn)

if (THREADING STREQUAL "TBB")
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
elseif (THREADING STREQUAL "OMP")
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_OMP)
else()
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_SEQ)
endif ()

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
)

if(ENABLE_MKL_DNN_JIT)
    add_definitions(-DMKLDNN_JIT=1)
    include_directories(
        ${MKLDNN_ROOT}/src/cpu
        ${MKLDNN_ROOT}/src/cpu/jit
        ${MKLDNN_ROOT}/src/cpu/jit/xbyak
        ${MKLDNN_ROOT}/src/cpu/gemm
        ${MKLDNN_ROOT}/src/cpu/gemm/jit
        ${MKLDNN_ROOT}/src/cpu/gemm/s8x8s32
        ${MKLDNN_ROOT}/src/cpu/gemm/f32
    )
else()
    foreach (ITEM ${SRC})
        if ("${ITEM}" MATCHES "(.*)/jit/(.*)")
            list(REMOVE_ITEM SRC ${ITEM})
        endif()
    endforeach()
endif()

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qlong-double /bigobj")
    endif()
endif()

if(THREADING STREQUAL "OMP")
    enable_omp()
endif()

add_library(${TARGET} STATIC ${HDR} ${SRC})
if(GEMM STREQUAL "OPENBLAS")
    ## enable cblas_gemm from OpenBLAS package
    add_definitions(-DUSE_CBLAS)
    include_directories(${BLAS_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${BLAS_LIBRARIES})
elseif (GEMM STREQUAL "MKL")
    ## enable cblas_gemm from mklml package
    include(MKL.cmake)
endif()
## enable internal jit_gemm from mkl-dnn if neither MKL nor OPENBLAS defined

target_link_libraries(${TARGET} ${${TARGET}_LINKER_LIBS})

