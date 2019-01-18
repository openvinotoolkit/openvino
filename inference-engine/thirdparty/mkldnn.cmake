#===============================================================================
# Copyright (c) 2016 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
#
#  Brief description: This cmake file replase original mkl-dnn build scripts
#  for more convenient integration to IE build process
#
#===============================================================================

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

function(detect_mkl LIBNAME)
    message(STATUS "Detecting Intel(R) MKL: trying ${LIBNAME}")
    find_path(MKLINC mkl_cblas.h ${MKL}/include)
    find_library(MKLLIB ${LIBNAME} "${MKL}/lib")

    if(NOT MKLLIB OR NOT MKLINC)
        message(FATAL_ERROR "${MKLINC} or ${MKLLIB} are not found")
        return()
    endif()

    if(WIN32)
        find_file(MKLDLL ${LIBNAME}.dll PATHS "${MKL}/lib")
        if(NOT MKLDLL)
            message(FATAL_ERROR "${LIBNAME} not found")
            return()
        endif()
    endif()

    set(MKLINC ${MKLINC} PARENT_SCOPE)
    set(MKLLIB "${MKLLIB}" PARENT_SCOPE)
    message(STATUS "Intel(R) MKL: include ${MKLINC}")
    message(STATUS "Intel(R) MKL: lib ${MKLLIB}")

    if(WIN32)
        set(MKLDLL "${MKLDLL}" PARENT_SCOPE)
        message(STATUS "Intel(R) MKL: dll ${MKLDLL}")
    endif()
endfunction()

set(TARGET mkldnn)
set(MKLDNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/mkl-dnn)

if (THREADING STREQUAL "TBB")
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_TBB)
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
        ${MKLDNN_ROOT}/src/cpu/xbyak
)

if(WIN32)
    add_definitions(-D_WIN)
    add_definitions(-DNOMINMAX)
    # Correct 'jnl' macro/jit issue
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qlong-double /bigobj")
    endif()
endif()

add_library(${TARGET} STATIC ${HDR} ${SRC})
set_ie_threading_interface_for(${TARGET})

if(GEMM STREQUAL "OPENBLAS")
    ## enable cblas_gemm from OpenBLAS package
    add_definitions(-DUSE_CBLAS)
    include_directories(${BLAS_INCLUDE_DIRS})
    list(APPEND ${TARGET}_LINKER_LIBS ${BLAS_LIBRARIES})
elseif (GEMM STREQUAL "MKL")
    ## enable cblas_gemm from mlkml package
    if(THREADING STREQUAL "TBB")
        detect_mkl("mkl_tiny_tbb")
    elseif (THREADING STREQUAL "OMP")
        detect_mkl("mkl_tiny_omp")
    else()
        detect_mkl("mkl_tiny_seq")
    endif()

    add_definitions(-DUSE_MKL -DUSE_CBLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND ${TARGET}_LINKER_LIBS ${MKLLIB})
endif()
## enable jit_gemm from mlk-dnn

target_link_libraries(${TARGET} PRIVATE ${${TARGET}_LINKER_LIBS})