# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)
if(ENABLE_DOCKER)
    cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
else()
    if (APPLE)
        # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
        cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
    else()
        cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
    endif()
endif()

project(OpenVINO)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)
include(features)

# include developer package
include(developer_package)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features)

# resolving dependencies for the project
message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "OpenVINO_MAIN_SOURCE_DIR .............. " ${OpenVINO_MAIN_SOURCE_DIR})
if (ENABLE_INFERENCE_ENGINE)
    set(IE_MAIN_SOURCE_DIR ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine)
    message (STATUS "IE_MAIN_SOURCE_DIR .............. " ${IE_MAIN_SOURCE_DIR})
endif()
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/targets_developer.cmake")

function(build_ngraph)
    if (NOT ENABLE_NGRAPH)
        return()
    endif()
    add_definitions(-DENABLE_NGRAPH)
    set(NGRAPH_BUILD_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} CACHE STRING "" FORCE)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/ngraph/cmake/Modules/")

    set(NGRAPH_ADDRESS_SANITIZER FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_UNIT_TEST_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_TOOLS_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_CPU_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_MLIR_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_INTELGPU_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_GPU_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_INTERPRETER_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_NOP_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_GPUH_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_GENERIC_CPU_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_DEBUG_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_DEPRECATED_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_ONNX_IMPORT_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_DEX_ONLY FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_ENABLE_CPU_CONV_AUTO FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_CODE_COVERAGE_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_LIB_VERSIONING_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_PYTHON_BUILD_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_PLAIDML_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_DISTRIBUTED_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_FAST_MATH_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_JSON_ENABLE FALSE CACHE BOOL "" FORCE)
    if (APPLE AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
        set(NGRAPH_STATIC_LIB_ENABLE TRUE CACHE BOOL "" FORCE)
    else()
        set(NGRAPH_STATIC_LIB_ENABLE FALSE CACHE BOOL "" FORCE)
    endif()
    set(NGRAPH_INTERPRETER_STATIC_LIB_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_CPU_STATIC_LIB_ENABLE FALSE CACHE BOOL "" FORCE)
    set(NGRAPH_DYNAMIC_COMPONENTS_ENABLE FALSE CACHE BOOL "" FORCE)

    check_cxx_compiler_flag("-Wmaybe-uninitialized" HAS_MAYBE_UNINITIALIZED)
    if (HAS_MAYBE_UNINITIALIZED)
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-uninitialized")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized -Wno-return-type")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized -Wno-return-type")
        endif()
    endif()
    # WA for GCC 7.0
    if (UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-return-type")
    elseif(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4308 /wd4146")
    endif()

    if(UNIX)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fvisibility=default")
    endif()

    if(ENABLE_LTO)
        ie_enable_lto()
    endif()

    add_subdirectory(ngraph)
    set(NGRAPH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ngraph/src/"
                           "${CMAKE_CURRENT_SOURCE_DIR}/ngraph/src/ngraph/" PARENT_SCOPE)
    set(NGRAPH_LIBRARY ngraph PARENT_SCOPE)

    export(TARGETS ngraph NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")
    export(TARGETS ngraph NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
endfunction()

build_ngraph()

if (ENABLE_INFERENCE_ENGINE)
    add_subdirectory(inference-engine)
endif()
