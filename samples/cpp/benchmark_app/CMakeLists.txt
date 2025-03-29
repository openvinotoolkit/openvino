# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "benchmark_app")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ov_add_sample(NAME ${TARGET_NAME}
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES ${GFLAGS_TARGET} format_reader ie_samples_utils)

# Required nlohmann_json dependency

if(NOT TARGET nlohmann_json::nlohmann_json)
    if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/nlohmann_json")
        # OpenVINO package puts thirdparty to samples dir
        # suppress shadowing names warning
        set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
        add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/nlohmann_json"
                            "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/json/nlohmann_json")
        # Allow running samples CMakeLists.txt as stand alone from openvino sources
        # suppress shadowing names warning
        set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
        add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/json/nlohmann_json"
                            "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "Failed to find / build nlohmann_json library")
    endif()
endif()

target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json::nlohmann_json)

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
        message(WARNING "OpenCL is disabled or not found, ${TARGET_NAME} will be built without OpenCL support. Install OpenCL.")
    endif()

    set(opencl_header_search_params
        HINTS
            ${opencl_root_hints}
        PATHS
            ENV "PROGRAMFILES(X86)"
            ENV AMDAPPSDKROOT
            ENV INTELOCLSDKROOT
            ENV NVSDKCOMPUTE_ROOT
            ENV CUDA_PATH
            ENV ATISTREAMSDKROOT
            ENV OCL_ROOT
        PATH_SUFFIXES
            "include"
            "OpenCL/common/inc"
            "AMD APP/include")

    find_path(OpenCL_HPP_INCLUDE_DIR
        NAMES
            CL/opencl.hpp OpenCL/opencl.hpp
        ${opencl_header_search_params})

    find_path(CL2_HPP_INCLUDE_DIR
        NAMES
            CL/cl2.hpp OpenCL/cl2.hpp
        ${opencl_header_search_params})

    if(OpenCL_FOUND AND (OpenCL_HPP_INCLUDE_DIR OR CL2_HPP_INCLUDE_DIR))
        set(OpenCL_DEFINITIONS HAVE_GPU_DEVICE_MEM_SUPPORT)

        # Append OpenCL CPP headers to C headers and use both
        if(OpenCL_HPP_INCLUDE_DIR)
            list(APPEND OpenCL_HEADERS ${OpenCL_HPP_INCLUDE_DIR})
            # the macro below is defined when opencl.hpp is found to suppress deprecation message from cl2.hpp
            list(APPEND OpenCL_DEFINITIONS OV_GPU_USE_OPENCL_HPP)
        endif()
        if(CL2_HPP_INCLUDE_DIR)
            list(APPEND OpenCL_HEADERS ${CL2_HPP_INCLUDE_DIR})
        endif()

        # cmake cannot set properties for imported targets
        get_target_property(opencl_target OpenCL::OpenCL ALIASED_TARGET)
        if(NOT TARGET ${opencl_target})
            set(opencl_target OpenCL::OpenCL)
        endif()

        set_property(TARGET ${opencl_target} APPEND PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_HEADERS})
        set_property(TARGET ${opencl_target} APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS ${OpenCL_DEFINITIONS})

        target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
    else()
        message(WARNING "OpenCL CPP header is not found, ${TARGET_NAME} will be built without OpenCL support and you will not be able to use the '-use_device_mem' option. Please, install '<apt | yum> install opencl-headers' to enable the option")
    endif()
endif()

# Optional OpenCV dependency

find_package(OpenCV QUIET COMPONENTS core)
if(NOT OpenCV_FOUND OR NOT OpenCV_VERSION VERSION_GREATER_EQUAL 3)
    message(WARNING "OpenCV ver. 3.0+ is not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()
