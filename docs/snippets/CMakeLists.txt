# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ie_docs_snippets)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ie_add_compiler_flags(-Wno-unused-variable)
    if(CMAKE_COMPILER_IS_GNUCXX)
        ie_add_compiler_flags(-Wno-unused-variable -Wno-unused-but-set-variable)
    endif()
endif()

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/vpu/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")

find_package(OpenCL)
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
endif()

# remove GPU remote snippets if OCL hasn't been found
if (NOT (OpenCL_FOUND AND OpenCL_HEADERS))
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/context_sharing_va.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/gpu/context_sharing.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/gpu/preprocessing.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/gpu/queue_sharing.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/gpu/remote_objects_creation.cpp")
endif()

# try to find VA libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(LIBVA QUIET libva)
endif()

# TODO: pkg_search_module finds libva not in sysroot
if(ANDROID)
    set(LIBVA_FOUND OFF CACHE BOOL "" FORCE)
endif()

# remove OpenCV related sources
if (ENABLE_OPENCV)
    find_package(OpenCV QUIET)
else()
    set(OpenCV_FOUND OFF)
endif()

if(NOT OpenCV_FOUND)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide5.cpp"
                             "${CMAKE_CURRENT_SOURCE_DIR}/ShapeInference.cpp")
endif()

# requires mfxFrameSurface1 and MSS API
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide2.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide3.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide4.cpp")

# create a static library

add_library(${TARGET_NAME} STATIC ${SOURCES})
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../template_extension/new/")

if(OpenCL_FOUND AND OpenCL_HEADERS)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIB})
    target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_HEADERS})

    if(LIBVA_FOUND)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
        target_include_directories(${TARGET_NAME} PRIVATE ${LIBVA_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} PRIVATE ${LIBVA_LINK_LIBRARIES})
    endif()

    if(WIN32)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d11 dxgi)
    endif()
endif()

if(OpenCV_FOUND)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    target_link_libraries(${TARGET_NAME} PRIVATE openvino_onnx_frontend)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::runtime::dev)

# ov_ncc_naming_style(FOR_TARGET "${TARGET_NAME}"
#                     SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                     ADDITIONAL_INCLUDE_DIRECTORIES
#                         $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>)

#
# Example
#

set(TARGET_NAME "ov_integration_snippet")
# [cmake:integration_example]
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)

add_executable(${TARGET_NAME} src/main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
# [cmake:integration_example]
