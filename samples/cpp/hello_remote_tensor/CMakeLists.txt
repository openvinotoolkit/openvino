# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "hello_remote_tensor")
ie_add_sample(NAME ${TARGET_NAME}
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ie_samples_utils)

# Optional OpenCL dependnency

if(DEFINED ENABLE_INTEL_GPU AND NOT ENABLE_INTEL_GPU)
    # Intel GPU plugin is turned off explicitly
    option(SAMPLES_ENABLE_OPENCL "Use OpenCL in benchmark_app" OFF)
else()
    option(SAMPLES_ENABLE_OPENCL "Use OpenCL in benchmark_app" ON)
endif()

if(SAMPLES_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND)
        MESSAGE(WARNING "OpenCL is disabled or not found, ${TARGET_NAME} will be built without OpenCL support. Install OpenCL.")
    endif()

    find_path(OpenCL_HPP_INCLUDE_DIR
        NAMES
        CL/opencl.hpp OpenCL/opencl.hpp
        HINTS
        ${opencl_root_hints}
        ENV "PROGRAMFILES(X86)"
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV NVSDKCOMPUTE_ROOT
        ENV CUDA_PATH
        ENV ATISTREAMSDKROOT
        ENV OCL_ROOT
        PATH_SUFFIXES
        include
        OpenCL/common/inc
        "AMD APP/include")

    find_path(CL2_HPP_INCLUDE_DIR
        NAMES
        CL/cl2.hpp OpenCL/cl2.hpp
        HINTS
        ${opencl_root_hints}
        ENV "PROGRAMFILES(X86)"
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV NVSDKCOMPUTE_ROOT
        ENV CUDA_PATH
        ENV ATISTREAMSDKROOT
        ENV OCL_ROOT
        PATH_SUFFIXES
        include
        OpenCL/common/inc
        "AMD APP/include")

    if(TARGET OpenCL)
        # Use OpenCL CPP headers from sources if present
        set(OpenCL_HEADERS ${OPENCL_HEADERS_DIR})
        set(OpenCL_LIB "OpenCL")
    elseif(OpenCL_HPP_INCLUDE_DIR OR CL2_HPP_INCLUDE_DIR)
        # Append OpenCL CPP headers to C headers and use both
        set(OpenCL_HEADERS ${OpenCL_INCLUDE_DIR})
        if (OpenCL_HPP_INCLUDE_DIR)
            list(APPEND ${OpenCL_HEADERS} ${OpenCL_HPP_INCLUDE_DIR})
        endif()
        if (CL2_HPP_INCLUDE_DIR)
            list(APPEND ${OpenCL_HEADERS} ${CL2_HPP_INCLUDE_DIR})
        endif()
        set(OpenCL_LIB "OpenCL::OpenCL")
    else()
        MESSAGE(WARNING "OpenCL CPP header is not found, ${TARGET_NAME} will be built without OpenCL support. Download it from: https://github.com/KhronosGroup/OpenCL-CLHPP and set -Dopencl_root_hints=[PATH]/OpenCL-CLHPP/include to cmake.")
    endif()

    if(OpenCL_FOUND AND OpenCL_HEADERS)
        target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIB})
        target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_HEADERS})
        target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_GPU_DEVICE_MEM_SUPPORT)
        if (OpenCL_HPP_INCLUDE_DIR OR OPENCL_HEADERS_DIR)
            # the macro below is defined when opencl.hpp is found to suppress deprecation message from cl2.hpp
            target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_USE_OPENCL_HPP)
        endif()
    endif()
endif()