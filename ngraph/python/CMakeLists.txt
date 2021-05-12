# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
    GIT_TAG        "v2.5.0"
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()


if(OpenVINO_MAIN_SOURCE_DIR)
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/)
    endif()
    
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY})
endif()

# compile options

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # disable warning: This operator was deprecated and will be removed with v0 operation.
    add_compile_options(/wd4996)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-deprecated-register -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_link_options(-stdlib=libc++)
    add_compile_options(-Wno-unused-value -Wno-range-loop-analysis)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # WA for GCC 7.5 "PYBIND11_NOINLINE inline" warning
    add_compile_options(-Wno-error=attributes)
endif()

# configure targets

add_subdirectory (src/pyopenvino)
add_subdirectory (src/pyngraph)
