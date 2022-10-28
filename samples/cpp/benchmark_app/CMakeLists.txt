# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "benchmark_app")

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME ${TARGET_NAME}
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES ${GFLAGS_TARGET} format_reader ie_samples_utils)

# Required nlohmann_json dependency

if(NOT ANDROID)
    find_package(PkgConfig QUIET)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json QUIET
                 # exception for Ubuntu 18.04, where cmake files for nlohmann_json
                 # are located in a wrong directory
                 HINTS /usr/lib/cmake)
    if(TARGET nlohmann_json)
        # Ubuntu 18.04 case where target 'nlohmann_json' is here, but nlohmann_json_FOUND is OFF
        if(NOT TARGET nlohmann_json::nlohmann_json)
            add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
        endif()
        set(nlohmann_json_FOUND ON)
    endif()
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

    if(NOT nlohmann_json_FOUND)
        # try to find header file json.hpp
        # for example, on debian 9 there is no cmake / pkgconfig files
        find_file(nlohmann_include_file
                  NAMES "json.hpp"
                  "Path to json.hpp (nlohmann-json-dev )")
        if(nlohmann_include_file)
            add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
            get_filename_component(nlohmann_include_dir "${nlohmann_include_file}" PATH)
            set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
                                  INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_include_dir}"
                                  INTERFACE_COMPILE_DEFINITIONS JSON_HEADER)
            set(nlohmann_json_FOUND ON)
        endif()
    endif()

    if(NOT nlohmann_json_FOUND)
        if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/nlohmann_json")
            # OpenVINO package puts thirdparty to samples dir
            # suppress shadowing names warning
            set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
            add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/nlohmann_json"
                             "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
        elseif(EXISTS "${Samples_SOURCE_DIR}../../thirdparty/nlohmann_json")
            # Allow running samples CMakeLists.txt as stand alone from openvino sources
            # suppress shadowing names warning
            set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
            add_subdirectory("${Samples_SOURCE_DIR}../../thirdparty/nlohmann_json"
                             "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
        else()
            message(FATAL_ERROR "Failed to find / build nlohmann_json library")
        endif()
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

# Optional OpenCV dependency

find_package(OpenCV QUIET COMPONENTS core)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core opencv_imgcodecs)
endif()
