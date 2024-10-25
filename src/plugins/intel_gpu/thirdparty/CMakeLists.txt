# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# oneDNN for GPU plugin
#

if(ENABLE_ONEDNN_FOR_GPU)
    function(build_onednn_gpu)
        include(ExternalProject)
        set(ONEDNN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_build")
        set(ONEDNN_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_install" CACHE PATH "Installation path for oneDNN GPU library")
        set(ONEDNN_PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_root")
        set(ONEDNN_ENABLED_PRIMITIVES "CONCAT;CONVOLUTION;DECONVOLUTION;INNER_PRODUCT;MATMUL;REORDER;POOLING;REDUCTION;SDPA")
        set(ONEDNN_ENABLED_ISA "XEHPG;XEHPC;XE2;XE3")
        set(DNNL_GPU_LIBRARY_NAME "openvino_onednn_gpu" CACHE STRING "Name of oneDNN library for Intel GPU Plugin")

        if(X86_64)
            set(ONEDNN_TARGET_ARCH "X64" CACHE STRING "" FORCE)
        elseif(X86)
            set(ONEDNN_TARGET_ARCH "X86" CACHE STRING "" FORCE)
        elseif(RISCV64)
            set(ONEDNN_TARGET_ARCH "RV64" CACHE STRING "" FORCE)
        elseif(ARM)
            set(ONEDNN_TARGET_ARCH "ARM" CACHE STRING "" FORCE)
        elseif(AARCH64)
            set(ONEDNN_TARGET_ARCH "AARCH64" CACHE STRING "" FORCE)
        else()
            message(WARNING "Intel GPU plugin unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
        endif()

        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
            ov_add_compiler_flags(-Wno-undef)
            ov_add_compiler_flags(-Wno-missing-declarations)
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
                ov_add_compiler_flags(-Wno-array-bounds)
                ov_add_compiler_flags(-Wno-stringop-overflow)
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
                    ov_add_compiler_flags(-Wno-restrict)
                endif()
            endif()
        endif()
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
        endif()

        foreach(cmake_var IN ITEMS CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION
                                   CMAKE_SYSTEM_PROCESSOR CMAKE_TOOLCHAIN_FILE)
            if(${cmake_var})
                list(APPEND cmake_extra_args "-D${cmake_var}=${${cmake_var}}")
            endif()
        endforeach()

        if(OV_GENERATOR_MULTI_CONFIG)
            if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
                list(APPEND cmake_extra_args "-DCMAKE_CONFIGURATION_TYPES=${CMAKE_DEFAULT_BUILD_TYPE}")
                list(APPEND cmake_extra_args "-DCMAKE_DEFAULT_BUILD_TYPE=${CMAKE_DEFAULT_BUILD_TYPE}")
            endif()
        else()
            list(APPEND cmake_extra_args "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
        endif()

        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
            if(DEFINED CMAKE_CXX_LINKER_LAUNCHER)
                list(APPEND cmake_extra_args "-DCMAKE_CXX_LINKER_LAUNCHER=${CMAKE_CXX_LINKER_LAUNCHER}")
            endif()
            if(DEFINED CMAKE_C_LINKER_LAUNCHER)
                list(APPEND cmake_extra_args "-DCMAKE_C_LINKER_LAUNCHER=${CMAKE_C_LINKER_LAUNCHER}")
            endif()
        endif()

        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15 AND DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
            list(APPEND cmake_extra_args "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}")
        endif()

        # propogate OpenCL if explicitly specified
        if(OpenCL_LIBRARY)
            list(APPEND cmake_extra_args "-DOpenCL_LIBRARY=${OpenCL_LIBRARY}")
        endif()
        if(OpenCL_INCLUDE_DIR)
            list(APPEND cmake_extra_args "-DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}")
        endif()

        set(onednn_gpu_lib "${CMAKE_STATIC_LIBRARY_PREFIX}${DNNL_GPU_LIBRARY_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(ONEDNN_GPU_LIB_PATH ${ONEDNN_INSTALL_DIR}/lib/${onednn_gpu_lib} CACHE FILEPATH "Path to oneDNN GPU library")

        # remove CMAKE_COMPILE_WARNING_AS_ERROR for onednn_gpu
        if(WIN32 AND CMAKE_COMPILE_WARNING_AS_ERROR AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_VERSION VERSION_LESS 3.24)
            ov_add_compiler_flags(/WX-)
        endif()

        ExternalProject_Add(onednn_gpu_build
            # Directory Options:
            PREFIX "${ONEDNN_PREFIX_DIR}"
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu"
            BINARY_DIR "${ONEDNN_BUILD_DIR}"
            INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
            # Configure Step Options:
            CMAKE_ARGS
                ${cmake_extra_args}
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
                "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
                "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=${ENABLE_LTO}"
                "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW"
                "-DDNNL_TARGET_ARCH=${ONEDNN_TARGET_ARCH}"
                "-DDNNL_CPU_RUNTIME=NONE"
                "-DDNNL_GPU_RUNTIME=OCL"
                "-DDNNL_LIBRARY_NAME=${DNNL_GPU_LIBRARY_NAME}"
                "-DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}"
                "-DDNNL_ENABLE_CONCURRENT_EXEC=ON"
                "-DDNNL_ENABLE_PRIMITIVE_CACHE=ON"
                "-DDNNL_ENABLE_WORKLOAD=INFERENCE"
                "-DDNNL_ENABLE_JIT_PROFILING=${BUILD_SHARED_LIBS}"
                "-DDNNL_ENABLE_ITT_TASKS=${BUILD_SHARED_LIBS}"
                "-DDNNL_BUILD_TESTS=OFF"
                "-DDNNL_BUILD_EXAMPLES=OFF"
                "-DDNNL_BLAS_VENDOR=NONE"
                "-DDNNL_LIBRARY_TYPE=STATIC"
                "-DDNNL_EXPERIMENTAL_PROFILING=ON"
                "-DONEDNN_BUILD_GRAPH=OFF"
                # specifically for Conan, because it overrides CMAKE_PREFIX_PATH and oneDNN's FindOpenCL.cmake is ignored
                # Conan's FindOpenCL.cmake module does not set OpenCL_INCLUDE_DIRS, so we need to set it manually
                "-DOpenCL_INCLUDE_DIRS=$<TARGET_PROPERTY:OpenCL::OpenCL,INTERFACE_INCLUDE_DIRECTORIES>"
                # Conan calls cmake with default value for CMP0091, so we have to bypass it to oneDNN build
                # because we bypass conan_toolchain.cmake via CMAKE_TOOLCHAIN_FILE
                "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
            CMAKE_CACHE_ARGS
                # The arguments below requires list to be passed as argument
                # which doesn't work properly when passed to CMAKE_ARGS.
                # Thus we pass it via CMAKE_CACHE_ARGS
                "-DDNNL_ENABLE_PRIMITIVE:STRING=${ONEDNN_ENABLED_PRIMITIVES}"
                "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA:STRING=${ONEDNN_ENABLED_ISA}"
            # Build Step Options:
            BUILD_BYPRODUCTS ${ONEDNN_GPU_LIB_PATH}
            # Target Options:
            EXCLUDE_FROM_ALL ON
        )

        list(APPEND LIB_INCLUDE_DIRS ${ONEDNN_INSTALL_DIR}/include)
        list(APPEND LIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu/src)
        set(LIB_DEFINITIONS ENABLE_ONEDNN_FOR_GPU
                            DNNL_DLL
                            DNNL_DLL_EXPORTS
                            DNNL_ENABLE_CPU_ISA_HINTS
                            DNNL_ENABLE_MAX_CPU_ISA
                            DNNL_X64=1
                            NGEN_CPP11
                            NGEN_NEO_INTERFACE
                            NGEN_NO_OP_NAMES
                            NGEN_SAFE
                            NGEN_WINDOWS_COMPAT)
        add_library(onednn_gpu_tgt INTERFACE)
        set_target_properties(onednn_gpu_tgt PROPERTIES
            INTERFACE_LINK_LIBRARIES $<BUILD_INTERFACE:${ONEDNN_GPU_LIB_PATH}>
            INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${LIB_INCLUDE_DIRS}>"
            INTERFACE_COMPILE_DEFINITIONS "${LIB_DEFINITIONS}"
        )
        add_dependencies(onednn_gpu_tgt onednn_gpu_build)

        if(NOT BUILD_SHARED_LIBS)
            ov_install_static_lib(onednn_gpu_tgt ${OV_CPACK_COMP_CORE})

            # we need to install library explicitly and set_target_properties in OpenVINOConfig.cmake for 'onednn_gpu_tgt'
            # to point to installation location of this file
            install(FILES "${ONEDNN_GPU_LIB_PATH}"
                    DESTINATION ${OV_CPACK_ARCHIVEDIR}
                    COMPONENT ${OV_CPACK_COMP_CORE})
        endif()
    endfunction()
    build_onednn_gpu()
endif()
