# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.3)

# Set the project name
project (ie_python_api)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(ENABLE_CONDA_FOLDER "Create output folder with conda python bindings" OFF)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH intel64)
elseif(ARCH STREQUAL "i386")
    set(ARCH ia32)
endif()

if(DEFINED IE_MAIN_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine inference_engine_plugin_api)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    set(InferenceEngine_LIBRARIES IE::inference_engine IE::inference_engine_plugin_api)
endif()

if(UNIX)
    # cython generated files requires public visibility. Force visibility required.
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_C_VISIBILITY_PRESET default)
endif()

include (UseCython)

if(PYTHONINTERP_FOUND)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpretator was not found!")
endif()

if(ENABLE_CONDA_FOLDER)
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/Conda/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/Conda/${PYTHON_VERSION}/openvino)
    endif()
else()
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
    endif()
endif()

set (PYTHON_BRIDGE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (src/openvino/inference_engine)

# Check Cython version
if("${CYTHON_VERSION}" VERSION_LESS "0.29")
    message(FATAL_ERROR "OpenVINO Python API needs at least Cython version 0.29, found version ${CYTHON_VERSION}")
else()
    message(STATUS "Found Cython version ${CYTHON_VERSION}")
endif()

# install

ie_cpack_add_component(${PYTHON_VERSION} REQUIRED)

install(FILES requirements.txt
        DESTINATION python/${PYTHON_VERSION}
        COMPONENT ${PYTHON_VERSION})

install(PROGRAMS src/openvino/__init__.py
        DESTINATION python/${PYTHON_VERSION}/openvino
        COMPONENT ${PYTHON_VERSION})

ie_cpack(${PYTHON_VERSION})
