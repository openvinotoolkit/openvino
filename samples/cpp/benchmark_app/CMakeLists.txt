# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "benchmark_app")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME ${TARGET_NAME}
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ie_samples_utils)

# Required nlohmann_json dependency

if(NOT ANDROID)
    find_package(PkgConfig QUIET)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json QUIET)
    if(nlohmann_json_FOUND)
        message(STATUS "nlohmann_json (${nlohmann_json_VERSION}) is found at ${nlohmann_json_DIR}")
    elseif(PkgConfig_FOUND)
        pkg_search_module(nlohmann_json QUIET
                          IMPORTED_TARGET GLOBAL
                          nlohmann_json)
        if(nlohmann_json_FOUND)
            add_library(nlohmann_json::nlohmann_json ALIAS PkgConfig::nlohmann_json)
        endif()
    endif()

    if(NOT nlohmann_json_FOUND AND EXISTS "${Samples_SOURCE_DIR}/thirdparty/nlohmann_json")
        # suppress shadowing names warning
        set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
        add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/nlohmann_json"
                         "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
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
        MESSAGE(WARNING "OpenCL is disabled or not found, ${TARGET_NAME} will be built without OpenCL support. Install OpenCL.")
    endif()

    find_path(OpenCL_HPP_INCLUDE_DIR
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
    elseif(OpenCL_HPP_INCLUDE_DIR)
        # Append OpenCL CPP headers to C headers and use both
        set(OpenCL_HEADERS ${OpenCL_INCLUDE_DIR} ${OpenCL_HPP_INCLUDE_DIR})
        set(OpenCL_LIB "OpenCL::OpenCL")
    else()
        MESSAGE(WARNING "OpenCL CPP header is not found, ${TARGET_NAME} will be built without OpenCL support. Download it from: https://github.com/KhronosGroup/OpenCL-CLHPP and set -Dopencl_root_hints=[PATH]/OpenCL-CLHPP/include to cmake.")
    endif()

    if(OpenCL_FOUND AND OpenCL_HEADERS)
        target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIB})
        target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_HEADERS})
        target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_GPU_DEVICE_MEM_SUPPORT)
    endif()
endif()

# Optional OpenCV dependency

find_package(OpenCV QUIET COMPONENTS core)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()
