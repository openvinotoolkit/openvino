# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)
if (APPLE)
    # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
    cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
endif()

project(OpenVINO)

set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_MODULE_PATH "${OpenVINO_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

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
file(REMOVE "${CMAKE_BINARY_DIR}/targets.cmake")

function(build_ngraph)
    if(NOT ENABLE_NGRAPH)
        return()
    endif()

    function(ngraph_set option value)
        if(NOT DEFINED ${option})
            set(${option} ${value} CACHE BOOL "" FORCE)
        endif()
    endfunction()

    add_definitions(-DENABLE_NGRAPH)
    set(NGRAPH_BUILD_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} CACHE STRING "" FORCE)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OpenVINO_MAIN_SOURCE_DIR}/ngraph/cmake/Modules/")

    ngraph_set(NGRAPH_ADDRESS_SANITIZER FALSE)
    ngraph_set(NGRAPH_TOOLS_ENABLE FALSE)
    ngraph_set(NGRAPH_CPU_ENABLE FALSE)
    ngraph_set(NGRAPH_MLIR_ENABLE FALSE)
    ngraph_set(NGRAPH_INTELGPU_ENABLE FALSE)
    ngraph_set(NGRAPH_GPU_ENABLE FALSE)
    ngraph_set(NGRAPH_INTERPRETER_ENABLE FALSE)
    ngraph_set(NGRAPH_NOP_ENABLE FALSE)
    ngraph_set(NGRAPH_GPUH_ENABLE FALSE)
    ngraph_set(NGRAPH_GENERIC_CPU_ENABLE FALSE)
    ngraph_set(NGRAPH_DEBUG_ENABLE FALSE)
    ngraph_set(NGRAPH_DEPRECATED_ENABLE FALSE)
    ngraph_set(NGRAPH_DEX_ONLY FALSE)
    ngraph_set(NGRAPH_ENABLE_CPU_CONV_AUTO FALSE)
    ngraph_set(NGRAPH_CODE_COVERAGE_ENABLE FALSE)
    ngraph_set(NGRAPH_LIB_VERSIONING_ENABLE FALSE)
    if (ENABLE_PYTHON AND NOT WIN32)
        ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE TRUE)
    else()
        ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE FALSE)
    endif()
    ngraph_set(NGRAPH_PLAIDML_ENABLE FALSE)
    ngraph_set(NGRAPH_DISTRIBUTED_ENABLE FALSE)
    ngraph_set(NGRAPH_FAST_MATH_ENABLE FALSE)
    ngraph_set(NGRAPH_JSON_ENABLE FALSE)
    ngraph_set(NGRAPH_STATIC_LIB_ENABLE FALSE)
    ngraph_set(NGRAPH_INTERPRETER_STATIC_LIB_ENABLE FALSE)
    ngraph_set(NGRAPH_CPU_STATIC_LIB_ENABLE FALSE)
    ngraph_set(NGRAPH_DYNAMIC_COMPONENTS_ENABLE FALSE)
    ngraph_set(NGRAPH_NATIVE_ARCH_ENABLE FALSE)

    if (NOT ANDROID)
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE TRUE)
        ngraph_set(NGRAPH_UNIT_TEST_OPENVINO_ENABLE TRUE)
    else()
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE FALSE)
        ngraph_set(NGRAPH_UNIT_TEST_OPENVINO_ENABLE FALSE)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        ie_add_compiler_flags(-Wno-error=uninitialized -Wno-error=literal-conversion)
    elseif(UNIX)
        ie_add_compiler_flags(-Wno-error=maybe-uninitialized -Wno-error=return-type -fPIC)
    endif()
    if(ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=defaulted-function-deleted -Wno-error=unused-command-line-argument")
    endif()

    # WA for GCC 7.0
    if (UNIX)
        ie_add_compiler_flags(-Wno-error=return-type -Wno-undef)
    elseif(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4308 /wd4146")
    endif()

    if(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
    endif()

    if(ENABLE_LTO)
        ie_enable_lto()
    endif()

    ie_cpack_add_component(ngraph)

    add_subdirectory(ngraph)
endfunction()

build_ngraph()

if (ENABLE_INFERENCE_ENGINE)
    add_subdirectory(inference-engine)
endif()

# cpack

ie_cpack(${IE_CPACK_COMPONENTS_ALL})
