# Copyright (C) 2018-2022 Intel Corporation
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
        if(CMAKE_COMPILER_IS_GNUCXX)
            ie_add_compiler_flags(-Wno-undef -Wno-suggest-override)
            if(OV_COMPILER_IS_CONDA)
                ie_add_compiler_flags(-Wno-error=array-bounds -Wno-error=stringop-overflow= -Wno-error=unused-result)
            endif()
        endif()
        ExternalProject_Add(onednn_gpu_build
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn_gpu"
            BINARY_DIR "${ONEDNN_BUILD_DIR}"
            INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
            PREFIX "${ONEDNN_PREFIX_DIR}"
            EXCLUDE_FROM_ALL ON
            CMAKE_ARGS
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
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DDNNL_ENABLE_CONCURRENT_EXEC=ON"
                "-DDNNL_ENABLE_PRIMITIVE_CACHE=OFF"
                "-DDNNL_ENABLE_JIT_PROFILING=${BUILD_SHARED_LIBS}"
                "-DDNNL_ENABLE_ITT_TASKS=${BUILD_SHARED_LIBS}"
                "-DDNNL_BUILD_TESTS=OFF"
                "-DDNNL_BUILD_EXAMPLES=OFF"
                "-DDNNL_BLAS_VENDOR=NONE"
                "-DDNNL_LIBRARY_TYPE=STATIC"
                "-DOpenCL_LIBRARY=${OpenCL_LIBRARY}"
                "-DOpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}"
                "-DOPENCL_VERSION_2_2=${OPENCL_VERSION_2_2}"
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
