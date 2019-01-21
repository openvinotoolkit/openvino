#===============================================================================
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Manage OpenMP-related compiler flags
#===============================================================================

if(OpenMP_cmake_included)
    return()
endif()
set(OpenMP_cmake_included true)

include("cmake/Threading.cmake")
include("cmake/MKL.cmake")

if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # OSX Clang doesn't have OpenMP by default.
    # But we still want to build the library.
    set(_omp_severity "WARNING")
else()
    set(_omp_severity "FATAL_ERROR")
endif()


macro(forbid_link_compiler_omp_rt)
    if (NOT WIN32)
        set_if(OpenMP_C_FOUND CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OpenMP_C_FLAGS})
        set_if(OpenMP_CXX_FOUND CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OpenMP_CXX_FLAGS})
        if (NOT APPLE)
            set (CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
        endif()
    endif()
endmacro()

macro(use_intel_omp_rt)
    # fast return
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        return()
    endif()

    # Do not link with compiler-native OpenMP library if Intel MKL is present.
    # Rationale: Intel MKL comes with Intel OpenMP library which is compatible
    # with all libraries shipped with compilers that Intel MKL-DNN supports.
    if(HAVE_MKL)
        forbid_link_compiler_omp_rt()
        if (UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # For some reasons Clang ignores `-fopenmp=libiomp5` switch and
            # links against libomp.so anyways.
            # The workaround is to set the full path to libiomp5.so
            add_library(libiomp5 SHARED IMPORTED)
            set_property(TARGET libiomp5 PROPERTY IMPORTED_LOCATION "${MKLIOMP5LIB}")
            list(APPEND EXTRA_LIBS libiomp5)
        else()
            list(APPEND EXTRA_LIBS ${MKLIOMP5LIB})
        endif()
    else()
        if (MKLDNN_THREADING STREQUAL "OMP:INTEL")
            message(${_omp_severity} "Intel OpenMP runtime could not be found. "
                "Please either use OpenMP runtime that comes with the compiler "
                "(via -DMKLDNN_THREADING={OMP,OMP:COMP}), or "
                "install Intel MKL / Intel MKL-ML (e.g. scripts/prepare_mkl.sh)")
        endif()
    endif()
endmacro()

if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    add_definitions(/Qpar)
    add_definitions(/openmp)
    set(OpenMP_CXX_FOUND true)
elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    append(CMAKE_C_FLAGS "-Xclang -fopenmp")
    append(CMAKE_CXX_FLAGS "-Xclang -fopenmp")
    set(OpenMP_CXX_FOUND true)
    list(APPEND EXTRA_LIBS ${MKLIOMP5LIB})
else()
    find_package(OpenMP)
    #newer version for findOpenMP (>= v. 3.9)
    if(CMAKE_VERSION VERSION_LESS "3.9" AND OPENMP_FOUND)
        if(${CMAKE_MAJOR_VERSION} VERSION_LESS "3" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
            # Override FindOpenMP flags for Intel Compiler (otherwise deprecated)
            set(OpenMP_CXX_FLAGS "-fopenmp")
            set(OpenMP_C_FLAGS "-fopenmp")
        endif()
        set(OpenMP_C_FOUND true)
        set(OpenMP_CXX_FOUND true)
    endif()
    append_if(OpenMP_C_FOUND CMAKE_C_FLAGS "${OpenMP_C_FLAGS}")
    append_if(OpenMP_CXX_FOUND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

if (MKLDNN_THREADING MATCHES "OMP")
    if (OpenMP_CXX_FOUND)
        set_threading("OMP")
    else()
        message(${_omp_severity} "OpenMP library could not be found. "
            "Proceeding might lead to highly sub-optimal performance.")
    endif()

    if (MKLDNN_THREADING STREQUAL "OMP:COMP")
        set(MKLIOMP5LIB "")
        set(MKLIOMP5DLL "")
    else()
        use_intel_omp_rt()
    endif()
else()
    # Compilation happens with OpenMP to enable `#pragma omp simd`
    # but during linkage OpenMP dependency should be avoided
    forbid_link_compiler_omp_rt()
    return()
endif()

set_ternary(_omp_lib_msg MKLIOMP5LIB "${MKLIOMP5LIB}" "provided by compiler")
message(STATUS "OpenMP lib: ${_omp_lib_msg}")
if(WIN32)
    set_ternary(_omp_dll_msg MKLIOMP5DLL "${MKLIOMP5LIB}" "provided by compiler")
    message(STATUS "OpenMP dll: ${_omp_dll_msg}")
endif()
