# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

# TODO: for make instal / package we need to use 3.13.3 version because
# it allows to install targets created outside of current projects
# See https://blog.kitware.com/cmake-3-13-0-available-for-download/

if (APPLE)
    # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
    cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
else()
    cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
endif()


project(OpenVINO)

set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IE_MAIN_SOURCE_DIR ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine)
set(CMAKE_MODULE_PATH "${OpenVINO_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(CTest)
include(features)

# include developer package
include(developer_package NO_POLICY_SCOPE)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features)
include(dependencies)

# resolving dependencies for the project
message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "OpenVINO_MAIN_SOURCE_DIR .............. " ${OpenVINO_MAIN_SOURCE_DIR})
message (STATUS "IE_MAIN_SOURCE_DIR .............. " ${IE_MAIN_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
file(REMOVE "${CMAKE_BINARY_DIR}/targets.cmake")

function(build_ngraph)
    function(ngraph_set option value)
        if(NOT DEFINED ${option})
            set(${option} ${value} CACHE BOOL "" FORCE)
        endif()
    endfunction()

    set(NGRAPH_BUILD_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} CACHE STRING "" FORCE)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OpenVINO_MAIN_SOURCE_DIR}/ngraph/cmake/Modules/")

    if (ENABLE_SANITIZER)
        ngraph_set(NGRAPH_ADDRESS_SANITIZER TRUE)
    else ()
        ngraph_set(NGRAPH_ADDRESS_SANITIZER FALSE)
    endif ()
    ngraph_set(NGRAPH_TOOLS_ENABLE FALSE)
    ngraph_set(NGRAPH_CPU_ENABLE FALSE)
    ngraph_set(NGRAPH_INTERPRETER_ENABLE TRUE)
    ngraph_set(NGRAPH_NOP_ENABLE FALSE)
    ngraph_set(NGRAPH_GPUH_ENABLE FALSE)
    ngraph_set(NGRAPH_GENERIC_CPU_ENABLE FALSE)
    ngraph_set(NGRAPH_ENABLE_CPU_CONV_AUTO FALSE)
    ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE FALSE)
    ngraph_set(NGRAPH_PLAIDML_ENABLE FALSE)
    ngraph_set(NGRAPH_FAST_MATH_ENABLE FALSE)
    ngraph_set(NGRAPH_JSON_ENABLE FALSE)
    ngraph_set(NGRAPH_NATIVE_ARCH_ENABLE FALSE)

    if (NOT ANDROID)
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE TRUE)
        ngraph_set(NGRAPH_UNIT_TEST_OPENVINO_ENABLE TRUE)
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE TRUE)
    else()
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE FALSE)
        ngraph_set(NGRAPH_TEST_UTIL_ENABLE FALSE)
        ngraph_set(NGRAPH_IE_ENABLE FALSE)
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE FALSE)
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4308 /wd4146 /wd4703 /wd4244")
    endif()

    if(ENABLE_LTO)
        ie_enable_lto()
    endif()

    ie_cpack_add_component(ngraph)

    set(SDL_cmake_included ON)
    # set(NGRAPH_COMPONENT_PREFIX "deployment_tools/ngraph/")
    add_subdirectory(ngraph)
endfunction()

build_ngraph()

add_subdirectory(inference-engine)

# cpack

# install setupvars

ie_cpack_add_component(setupvars REQUIRED)

if(UNIX)
    install(PROGRAMS scripts/setupvars/setupvars.sh
            DESTINATION bin
            COMPONENT setupvars)
elseif(WIN32)
    install(PROGRAMS scripts/setupvars/setupvars.bat
            DESTINATION bin
            COMPONENT setupvars)
endif()

# install install_dependencies

if(UNIX)
    ie_cpack_add_component(install_dependencies REQUIRED)
    install(DIRECTORY scripts/install_dependencies/
            DESTINATION install_dependencies
            COMPONENT install_dependencies)
endif()

# install files for demo

ie_cpack_add_component(demo_scripts REQUIRED DEPENDS core)

if(UNIX)
    install(DIRECTORY scripts/demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE)
elseif(WIN32)
    install(DIRECTORY scripts/demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE)
endif()

ie_cpack(${IE_CPACK_COMPONENTS_ALL})
