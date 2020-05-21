#===============================================================================
# Copyright (C) 2018-2020 Intel Corporation
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

set(version_cmake_included true)

if(NOT TARGET)
    set(TARGET mkldnn)
endif()

set(MKLDNN_ROOT ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn)

string(REPLACE "." ";" VERSION_LIST "0.18.0")
list(GET VERSION_LIST 0 MKLDNN_VERSION_MAJOR)
list(GET VERSION_LIST 1 MKLDNN_VERSION_MINOR)
list(GET VERSION_LIST 2 MKLDNN_VERSION_PATCH)

find_package(Git)
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
            WORKING_DIRECTORY ${MKLDNN_ROOT}
            RESULT_VARIABLE RESULT
            OUTPUT_VARIABLE MKLDNN_VERSION_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR RESULT)
    set(MKLDNN_VERSION_HASH "N/A")
endif()

if(NOT EXISTS "${CMAKE_BINARY_DIR}/include/mkldnn_version.h")
    configure_file(
        "${MKLDNN_ROOT}/include/mkldnn_version.h.in"
        "${CMAKE_BINARY_DIR}/include/mkldnn_version.h"
    )
endif()

function(detect_mkl LIBNAME)
    unset(MKLLIB CACHE)
    unset(MKLINC CACHE)

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

if (THREADING STREQUAL "TBB")
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_TBB)
elseif (THREADING STREQUAL "TBB_AUTO")
    add_definitions(-DMKLDNN_THR=MKLDNN_THR_TBB_AUTO)
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
        ${MKLDNN_ROOT}/src/cpu/
        ${MKLDNN_ROOT}/src/cpu/xbyak
        ${CMAKE_BINARY_DIR}/include/
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
    if(WIN32 OR APPLE)
        detect_mkl("mklml")
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            detect_mkl("mklml_intel")
        else()
            detect_mkl("mklml_gnu")
        endif()
    endif()
    add_definitions(-DUSE_MKL -DUSE_CBLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND ${TARGET}_LINKER_LIBS ${MKLLIB})
endif()
## enable jit_gemm from mlk-dnn

add_definitions(-DMKLDNN_ENABLE_CONCURRENT_EXEC)

target_link_libraries(${TARGET} PRIVATE ${${TARGET}_LINKER_LIBS})