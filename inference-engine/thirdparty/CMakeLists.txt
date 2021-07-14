# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_MYRIAD)
    add_subdirectory(movidius)
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_GREATER_EQUAL "1910"))
    # 1910 version of Visual Studio 2017
    # This flagis needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

if (ENABLE_CLDNN)
    if (ENABLE_TESTS AND ENABLE_CLDNN_TESTS)
        set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "" FORCE)
    else()
        set(CLDNN__INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    endif()
    set(CLDNN_THREADING "${THREADING}" CACHE STRING "" FORCE)
    set(GPU_DEBUG_CONFIG OFF CACHE BOOL "Enable debug config feature")
    add_subdirectory(clDNN)
endif()

function(ie_add_mkldnn)
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE) ## TODO: try it later
    set(DNNL_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)     ## TODO: try it later
    set(DNNL_LIBRARY_TYPE STATIC CACHE BOOL "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE BOOL "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE BOOL "" FORCE)
    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX)
        ie_add_compiler_flags(-Wno-undef)
        if(SUGGEST_OVERRIDE_SUPPORTED)
            # xbyak compilation fails
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
        endif()
    endif()

    add_subdirectory(mkl-dnn EXCLUDE_FROM_ALL)
    add_library(mkldnn ALIAS dnnl)
endfunction()

if(ENABLE_MKL_DNN)
    ie_add_mkldnn()
endif()
