# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
    # 1910 version of Visual Studio 2017
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

function(ie_add_onednn)
    set(DNNL_ENABLE_JIT_PROFILING ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
    if(BUILD_SHARED_LIBS AND ENABLE_PROFILING_ITT)
        set(DNNL_ENABLE_ITT_TASKS ON CACHE BOOL "" FORCE)
    else()
        set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)
    endif()
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE) ## TODO: try it later
    set(DNNL_ENABLE_MAX_CPU_ISA ON CACHE BOOL "" FORCE)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE STRING "" FORCE)
    set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
    if (ENABLE_CPU_DEBUG_CAPS)
        set(DNNL_VERBOSE "ON" CACHE STRING "" FORCE)
    else()
        set(DNNL_VERBOSE "OFF" CACHE STRING "" FORCE)
    endif()

    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;DECONVOLUTION;CONCAT;LRN;INNER_PRODUCT;MATMUL;POOLING;REDUCTION;REORDER;RNN;SOFTMAX" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)

    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        # xbyak compilation fails
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX)
        ie_add_compiler_flags(-Wno-undef)
        ie_add_compiler_flags(-Wno-missing-declarations)
    elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10121")
    endif()

    add_subdirectory(onednn EXCLUDE_FROM_ALL)
    ov_install_static_lib(dnnl cpu)
endfunction()

if(ENABLE_INTEL_CPU)
    ie_add_onednn()
endif()
