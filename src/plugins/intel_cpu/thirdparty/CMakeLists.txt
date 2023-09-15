# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(intel_cpu_thirdparty)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND MSVC_TOOLSET_VERSION GREATER_EQUAL 141)
    # Visual Studio 2017 (v141 toolset)
    # This flag is needed for enabling SIMD vectorization with command '#pragma omp simd'.
    # Compilation with '/openmp:experimental' key allow us to enable vectorizatikon capability in MSVC.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

function(ov_add_onednn)
    set(CMAKE_DISABLE_FIND_PACKAGE_PythonInterp ON)
    set(DNNL_ENABLE_JIT_PROFILING ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
    if(BUILD_SHARED_LIBS AND ENABLE_PROFILING_ITT)
        set(DNNL_ENABLE_ITT_TASKS ON CACHE BOOL "" FORCE)
    else()
        set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)
    endif()
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE) # Enable primitive cache for global sharing
    set(DNNL_ENABLE_MAX_CPU_ISA ON CACHE BOOL "" FORCE)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE STRING "" FORCE)
    set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_GEMM_KERNELS_ISA "SSE41" CACHE STRING "" FORCE) #Only build gemm driver for SSE41 ISA.

    # plugin does not use onednn graph
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    # select needed primitives
    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;DECONVOLUTION;CONCAT;LRN;INNER_PRODUCT;MATMUL;POOLING;REDUCTION;REORDER;RNN;SOFTMAX" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
    set(DNNL_LIBRARY_NAME "openvino_onednn_cpu" CACHE STRING "" FORCE)

    # Allow to enable oneDNN verbose with CPU_DEBUG_CAPS and rely on oneDNN default configuration otherwise
    if(ENABLE_CPU_DEBUG_CAPS)
        set(DNNL_VERBOSE ON CACHE STRING "" FORCE)
    endif()

    if(X86_64)
        set(DNNL_TARGET_ARCH "X64" CACHE STRING "" FORCE)
    elseif(X86)
        set(DNNL_TARGET_ARCH "X86" CACHE STRING "" FORCE)
    elseif(RISCV64)
        set(DNNL_TARGET_ARCH "RV64" CACHE STRING "" FORCE)
    elseif(ARM)
        set(DNNL_TARGET_ARCH "ARM" CACHE STRING "" FORCE)
    elseif(AARCH64)
        set(DNNL_TARGET_ARCH "AARCH64" CACHE STRING "" FORCE)
    else()
        message(FATAL_ERROR "Unsupported system processor ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    if(AARCH64 OR ARM)
        set(DNNL_USE_ACL ON CACHE BOOL "Use ARM Conpute Library kernels in oneDNN" FORCE)
    endif()

    set(SDL_cmake_included ON)  ## to skip internal SDL flags. SDL flags are already set on IE level
    if (ANDROID OR ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR OV_COMPILER_IS_CLANG) AND NOT (THREADING STREQUAL "OMP")))
        set(OpenMP_cmake_included ON) ## to skip "omp simd" inside a code. Lead to some crashes inside NDK LLVM..
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

    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        ov_add_compiler_flags(-Wno-undef)
        ov_add_compiler_flags(-Wno-missing-declarations)
        if(ARM OR AARCH64)
            ov_add_compiler_flags(-Wno-macro-redefined)
        endif()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
            ov_add_compiler_flags(-Wno-array-bounds)
            ov_add_compiler_flags(-Wno-stringop-overflow)
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
                ov_add_compiler_flags(-Wno-restrict)
            endif()
        endif()
    elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10121")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # C4849 OpenMP 'reduction' clause ignored in 'simd' directive
        ov_add_compiler_flags(/wd4849)
        # C4661 no suitable definition provided for explicit template instantiation request
        ov_add_compiler_flags(/wd4661)
        # C4267, 4244 conversion from 'XXX' to 'YYY', possible loss of data
        ov_add_compiler_flags(/wd4267)
        ov_add_compiler_flags(/wd4244)
        # C4334 '<<': result of 32-bit shift implicitly converted to 64 bits
        ov_add_compiler_flags(/wd4334)
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        # xbyak compilation fails
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
    endif()

    # to find our FindACL.cmake
    if(DNNL_USE_ACL)
        list(APPEND CMAKE_MODULE_PATH "${intel_cpu_thirdparty_SOURCE_DIR}")
    endif()

    add_subdirectory(onednn EXCLUDE_FROM_ALL)

    # install static libraries
    ov_install_static_lib(dnnl ${OV_CPACK_COMP_CORE})

    if(DNNL_USE_ACL AND NOT BUILD_SHARED_LIBS)
        # use ACLConfig.cmake in OpenVINOConfig.cmake in case of static build
        # we cannot use 'ov_install_static_lib' for imported targets,
        # but for this we need to install library files
        install(FILES $<TARGET_PROPERTY:arm_compute::arm_compute,IMPORTED_LOCATION>
                DESTINATION ${OV_CPACK_ARCHIVEDIR}
                COMPONENT ${OV_CPACK_COMP_CORE})
        install(FILES "${intel_cpu_thirdparty_SOURCE_DIR}/ACLConfig.cmake"
                DESTINATION ${OV_CPACK_OPENVINO_CMAKEDIR}
                COMPONENT ${OV_CPACK_COMP_CORE_DEV})
    endif()
endfunction()

if(ENABLE_MLAS_FOR_CPU)
    add_subdirectory(mlas)
    ov_install_static_lib(mlas ${OV_CPACK_COMP_CORE})
endif()

ov_add_onednn()
