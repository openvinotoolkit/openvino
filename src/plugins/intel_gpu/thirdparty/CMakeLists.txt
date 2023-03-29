# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_library(rapidjson INTERFACE)

set_target_properties(rapidjson PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rapidjson>
)

ov_install_static_lib(rapidjson gpu)

#
# oneDNN for GPU plugin
#

if(ENABLE_ONEDNN_FOR_GPU)
    function(build_onednn_gpu)
        include(ExternalProject)
        set(ONEDNN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_build/")
        set(ONEDNN_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_install/")
        set(ONEDNN_PREFIX_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn_gpu_root")
        set(ONEDNN_ENABLED_PRIMITIVES "CONCAT;CONVOLUTION;DECONVOLUTION;INNER_PRODUCT;MATMUL;REORDER;POOLING;REDUCTION")
        set(ONEDNN_ENABLED_ISA "XEHP;XEHPG;XEHPC")

        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            ie_add_compiler_flags(-Wno-undef)
            ie_add_compiler_flags(-Wno-missing-declarations)
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
                ie_add_compiler_flags(-Wno-array-bounds)
                ie_add_compiler_flags(-Wno-stringop-overflow)
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
                    ie_add_compiler_flags(-Wno-restrict)
                endif()
            endif()
        endif()
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
        endif()

        foreach(cmake_var IN ITEMS CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION
                                   CMAKE_SYSTEM_PROCESSOR CMAKE_TOOLCHAIN_FILE)
            list(APPEND cmake_extra_args "-D${cmake_var}=${${cmake_var}}")
        endforeach()

        if(OV_GENERATOR_MULTI_CONFIG)
            if(CMAKE_GENERATOR MATCHES "^Ninja Multi-Config$")
                list(APPEND cmake_extra_args "-DCMAKE_CONFIGURATION_TYPES=${CMAKE_DEFAULT_BUILD_TYPE}")
                list(APPEND cmake_extra_args "-DCMAKE_DEFAULT_BUILD_TYPE=${CMAKE_DEFAULT_BUILD_TYPE}")
            endif()
        else()
            list(APPEND cmake_extra_args "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
        endif()

        ExternalProject_Add(onednn_gpu_build
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu"
            BINARY_DIR "${ONEDNN_BUILD_DIR}"
            INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
            PREFIX "${ONEDNN_PREFIX_DIR}"
            EXCLUDE_FROM_ALL ON
            CMAKE_CACHE_ARGS
                # The arguments below requires list to be passed as argument
                # which doesn't work properly when passed to CMAKE_ARGS.
                # Thus we pass it via CMAKE_CACHE_ARGS
                "-DDNNL_ENABLE_PRIMITIVE:STRING=${ONEDNN_ENABLED_PRIMITIVES}"
                "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA:STRING=${ONEDNN_ENABLED_ISA}"
            CMAKE_ARGS
                ${cmake_extra_args}
                "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
                "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
                "-DCMAKE_CXX_LINKER_LAUNCHER=${CMAKE_CXX_LINKER_LAUNCHER}"
                "-DCMAKE_C_LINKER_LAUNCHER=${CMAKE_C_LINKER_LAUNCHER}"
                "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=${ENABLE_LTO}"
                "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW"
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
                "-DDNNL_CPU_RUNTIME=NONE"
                "-DDNNL_GPU_RUNTIME=OCL"
                "-DDNNL_LIBRARY_NAME=onednn_gpu"
                "-DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}"
                "-DCMAKE_INSTALL_LIBDIR=lib/$<CONFIG>"
                "-DDNNL_ENABLE_CONCURRENT_EXEC=ON"
                "-DDNNL_ENABLE_PRIMITIVE_CACHE=ON"
                "-DDNNL_ENABLE_WORKLOAD=INFERENCE"
                "-DDNNL_ENABLE_JIT_PROFILING=${BUILD_SHARED_LIBS}"
                "-DDNNL_ENABLE_ITT_TASKS=${BUILD_SHARED_LIBS}"
                "-DDNNL_BUILD_TESTS=OFF"
                "-DDNNL_BUILD_EXAMPLES=OFF"
                "-DDNNL_BLAS_VENDOR=NONE"
                "-DDNNL_LIBRARY_TYPE=STATIC"
                "-DOpenCL_LIBRARY=${OpenCL_LIBRARY}"
                "-DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}"
        )
        add_library(onednn_gpu_tgt INTERFACE)
        set_target_properties(onednn_gpu_tgt PROPERTIES
            INTERFACE_LINK_DIRECTORIES "${ONEDNN_INSTALL_DIR}/lib/$<CONFIG>"
            INTERFACE_LINK_LIBRARIES "onednn_gpu"
            INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INSTALL_DIR}/include"
            INTERFACE_COMPILE_DEFINITIONS ENABLE_ONEDNN_FOR_GPU
        )
        add_dependencies(onednn_gpu_tgt onednn_gpu_build)
        # TODO: install onednn_gpu in static builds
    endfunction()
    build_onednn_gpu()
endif()
