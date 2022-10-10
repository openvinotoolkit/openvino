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

    # select needed primitives
    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;DECONVOLUTION;CONCAT;LRN;INNER_PRODUCT;MATMUL;POOLING;REDUCTION;REORDER;RNN;SOFTMAX" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)

    # Allow to enable oneDNN verbose with CPU_DEBUG_CAPS and rely on oneDNN default configuration otherwise
    if (ENABLE_CPU_DEBUG_CAPS)
        set(DNNL_VERBOSE "ON" CACHE STRING "" FORCE)
    endif()

    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        # xbyak compilation fails
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
    endif()
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(DNNL_TARGET_ARCH "X64")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(DNNL_TARGET_ARCH "AARCH64")
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        ie_add_compiler_flags(-Wno-undef)
        ie_add_compiler_flags(-Wno-missing-declarations)
        if(OV_COMPILER_IS_CONDA)
            ie_add_compiler_flags(-Wno-error=array-bounds)
            ie_add_compiler_flags(-Wno-error=stringop-overflow=)
        endif()
    elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10121")
    endif()

    # WA for old TBBConfig.cmake like tbb2019_20180718oss
    # they don't check that imported target is already created
    if(TBB_FOUND)
        set(TBB_cmake_included ON)
        set(DNNL_CPU_THREADING_RUNTIME "${THREADING}")
        function(find_package_tbb)
            # dummy
        endfunction()
        link_libraries(TBB::tbb)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # C4849 OpenMP 'reduction' clause ignored in 'simd' directive
        ie_add_compiler_flags(/wd4849)
        # C4661 no suitable definition provided for explicit template instantiation request
        ie_add_compiler_flags(/wd4661)
        # C4267, 4244 conversion from 'XXX' to 'YYY', possible loss of data
        ie_add_compiler_flags(/wd4267)
        ie_add_compiler_flags(/wd4244)
        # C4334 '<<': result of 32-bit shift implicitly converted to 64 bits
        ie_add_compiler_flags(/wd4334)
    endif()
    add_subdirectory(onednn EXCLUDE_FROM_ALL)
    ov_install_static_lib(dnnl cpu)
endfunction()

if(ENABLE_INTEL_CPU)
    ie_add_onednn()
endif()
