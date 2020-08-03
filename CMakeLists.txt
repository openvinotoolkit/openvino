# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

# TODO: for make instal / package we need to use 3.13.3 version because
# it allows to install targets created outside of current projects
# See https://blog.kitware.com/cmake-3-13-0-available-for-download/

if (APPLE)
    if(CMAKE_GENERATOR STREQUAL "Xcode")
        # due to https://gitlab.kitware.com/cmake/cmake/issues/14254
        cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
    else()
        # due to https://cmake.org/cmake/help/v3.12/policy/CMP0068.html
        cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
    endif()
else()
    cmake_minimum_required(VERSION 3.7.2 FATAL_ERROR)
endif()

project(OpenVINO)

set(OpenVINO_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IE_MAIN_SOURCE_DIR ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine)
list(APPEND CMAKE_MODULE_PATH "${OpenVINO_MAIN_SOURCE_DIR}/cmake")

include(CTest)
include(features)

# include developer package
include(developer_package)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features)
include(dependencies)

# resolving dependencies for the project
message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "OpenVINO_MAIN_SOURCE_DIR .............. " ${OpenVINO_MAIN_SOURCE_DIR})
message (STATUS "IE_MAIN_SOURCE_DIR .................... " ${IE_MAIN_SOURCE_DIR})
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
    ngraph_set(NGRAPH_PYTHON_BUILD_ENABLE FALSE)

    if(ENABLE_TESTS AND NOT ANDROID)
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE TRUE)
        ngraph_set(NGRAPH_IE_ENABLE TRUE)
    else()
        ngraph_set(NGRAPH_UNIT_TEST_ENABLE FALSE)
        ngraph_set(NGRAPH_IE_ENABLE FALSE)
    endif()

    if(NOT ANDROID)
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE TRUE)
    else()
        ngraph_set(NGRAPH_ONNX_IMPORT_ENABLE FALSE)
    endif()
    ngraph_set(NGRAPH_INTERPRETER_ENABLE TRUE)

    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4308 /wd4146 /wd4703 /wd4244 /wd4819")
    endif()
    
    # Preserve the original flags for further use
    set(CMAKE_ORIGINAL_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_ORIGINAL_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    set(CMAKE_ORIGINAL_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    set(CMAKE_ORIGINAL_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    set(CMAKE_ORIGINAL_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE}")

    if(ENABLE_LTO)
        ie_enable_lto()
    endif()

    ie_cpack_add_component(ngraph)

    set(SDL_cmake_included ON)
    # set(NGRAPH_COMPONENT_PREFIX "deployment_tools/ngraph/")
    add_subdirectory(ngraph)
    set(NGRAPH_LIBRARIES ngraph PARENT_SCOPE)
endfunction()

build_ngraph()

add_subdirectory(inference-engine)

add_subdirectory(docs)

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
