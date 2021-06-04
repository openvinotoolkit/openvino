# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.13)

# Set the project name
project (ie_python_api)

option(ENABLE_CONDA_FOLDER "Create output folder with conda python bindings" OFF)
option(ENABLE_WHEEL "Create wheel package" OFF)

set(PYTHON_BRIDGE_CPACK_PATH "python")

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH intel64)
elseif(ARCH STREQUAL "i386")
    set(ARCH ia32)
endif()

if(DEFINED IE_MAIN_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    set(InferenceEngine_LIBRARIES IE::inference_engine)
endif()

if(UNIX)
    # cython generated files requires public visibility. Force visibility required.
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_C_VISIBILITY_PRESET default)
endif()

include (cmake/UseCython.cmake)

if(Python3_Interpreter_FOUND)
    set(Python3_VERSION python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpretator was not found!")
endif()

if(ENABLE_CONDA_FOLDER)
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/Conda/${Python3_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/Conda/${Python3_VERSION}/openvino)
    endif()
else()
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${Python3_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${Python3_VERSION}/openvino)
    endif()
endif()

set (PYTHON_BRIDGE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (src/openvino/inference_engine)
add_subdirectory (src/openvino/offline_transformations)

if (ENABLE_TESTS)
    add_subdirectory(src/openvino/test_utils)
endif()

if(ENABLE_WHEEL)
    add_subdirectory(wheel)
endif()

# Check Cython version
if(CYTHON_VERSION VERSION_LESS "0.29")
    message(FATAL_ERROR "OpenVINO Python API needs at least Cython version 0.29, found version ${CYTHON_VERSION}")
else()
    message(STATUS "Found Cython version ${CYTHON_VERSION}")
endif()

# install

ie_cpack_add_component(${Python3_VERSION})

install(FILES requirements.txt
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${Python3_VERSION}
        COMPONENT ${Python3_VERSION})

install(FILES requirements.txt
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}
        COMPONENT ${Python3_VERSION})

install(PROGRAMS src/openvino/__init__.py
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${Python3_VERSION}/openvino
        COMPONENT ${Python3_VERSION})

# install Python samples

ie_cpack_add_component(python_samples)

install(DIRECTORY sample/
        DESTINATION ${IE_CPACK_IE_DIR}/samples/python
        COMPONENT python_samples)

ie_cpack(${Python3_VERSION} python_samples)
