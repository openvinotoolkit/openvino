# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Defines the CMake commands/policies
cmake_minimum_required (VERSION 3.3)

# Set the project name
project (ie_python_api)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH intel64)
elseif(ARCH STREQUAL "i386")
    set(ARCH ia32)
endif()

if(ENABLE_NGRAPH)
    add_definitions(-DENABLE_NGRAPH)
endif()

if(DEFINED IE_MAIN_SOURCE_DIR)
    set(InferenceEngine_LIBRARIES inference_engine)
else()
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

if(UNIX)
    # cython generated files requires public visibility. Force visibility required.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
endif()

include (UseCython)

if(PYTHONINTERP_FOUND)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpretator was not found!")
endif()

if(WIN32)
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
else()
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
endif()

set (PYTHON_BRIDGE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (src/openvino/inference_engine)

# install

ie_cpack_add_component(${PYTHON_VERSION} REQUIRED)

install(FILES requirements.txt
        DESTINATION python/${PYTHON_VERSION}
        COMPONENT ${PYTHON_VERSION})

install(PROGRAMS src/openvino/__init__.py
        DESTINATION python/${PYTHON_VERSION}/openvino
        COMPONENT ${PYTHON_VERSION})

ie_cpack(${PYTHON_VERSION})
