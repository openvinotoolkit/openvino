# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project (pyngraph)

if(NOT DEFINED OpenVINO_MAIN_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage QUIET)
    find_package(ngraph REQUIRED)
endif()

if(ngraph_FOUND)
    message("ngraph version = {${ngraph_VERSION}}")
endif()

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

# PYTHON_VERSION_MAJOR and PYTHON_VERSION_MINOR are defined inside pybind11
set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
message("Python version=${PYTHON_VERSION}")

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

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # for proper fix need to update pybind to version which does not use PyEval_InitThreads()
    add_compile_options(-Wno-deprecated-declarations -Wno-undef)
endif()

# create target

file(GLOB_RECURSE SOURCES src/pyngraph/*.cpp)

pybind11_add_module(_${PROJECT_NAME} MODULE ${SOURCES})

target_include_directories(_${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(_${PROJECT_NAME} PRIVATE ngraph::ngraph)
if (TARGET ngraph::onnx_importer)
    add_dependencies(_${PROJECT_NAME} ngraph::onnx_importer)
endif()

# perform copy
if(OpenVINO_MAIN_SOURCE_DIR)
    add_custom_command(TARGET _${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/ngraph ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ngraph
    )
endif()

if(OpenVINO_MAIN_SOURCE_DIR OR InferenceEngineDeveloperPackage_FOUND)
    if(COMMAND ie_python_minimal_api)
        ie_python_minimal_api(_${PROJECT_NAME})
    endif()

    add_clang_format_target(_${PROJECT_NAME}_clang FOR_TARGETS _${PROJECT_NAME})

    ie_cpack_add_component(pyngraph_${PYTHON_VERSION})

    install(TARGETS _${PROJECT_NAME}
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyngraph_${PYTHON_VERSION})

    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/ngraph
            DESTINATION python/${PYTHON_VERSION}
            COMPONENT pyngraph_${PYTHON_VERSION}
            USE_SOURCE_PERMISSIONS)

    ie_cpack(pyngraph_${PYTHON_VERSION})
endif()
