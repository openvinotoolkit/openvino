# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ie_docs_snippets)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ie_add_compiler_flags(-Wno-unused-variable)
endif()
if(UNUSED_BUT_SET_VARIABLE_SUPPORTED)
    ie_add_compiler_flags(-Wno-unused-but-set-variable)
endif()

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB GPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp")

# add GPU snippets if OpenCL has been found
if(TARGET OpenCL::OpenCL)
    list(APPEND SOURCES ${GPU_SOURCES})
endif()

# try to find VA libraries
if(NOT ANDROID)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_search_module(libva QUIET IMPORTED_TARGET libva)
        if(libva_FOUND)
            message(STATUS "${PKG_CONFIG_EXECUTABLE}: libva (${libva_VERSION}) is found at ${libva_PREFIX}")
        endif()
    endif()
endif()

# remove OpenCV related sources
find_package(OpenCV QUIET COMPONENTS core imgcodecs)
if(NOT OpenCV_FOUND)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ShapeInference.cpp")
endif()

# requires mfxFrameSurface1 and MSS API
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide2.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide3.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/dldt_optimization_guide4.cpp")

# create a static library

add_library(${TARGET_NAME} STATIC ${SOURCES})
target_include_directories(${TARGET_NAME} PRIVATE "${OpenVINO_SOURCE_DIR}/src/inference/include/ie"
                                                  "${OpenVINO_SOURCE_DIR}/src/inference/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/inference/dev_api"
                                                  "${OpenVINO_SOURCE_DIR}/src/core/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/bindings/c/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/common/transformations/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/common/preprocessing/src"
                                                  "${OpenVINO_SOURCE_DIR}/src/common/util/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/common/low_precision_transformations/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/frontends/common/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/core/template_extension/new"
                                                  "${OpenVINO_SOURCE_DIR}/src/frontends/onnx/frontend/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/frontends/tensorflow/include"
                                                  "${OpenVINO_SOURCE_DIR}/src/frontends/paddle/include")

ov_mark_target_as_cc(${TARGET_NAME})

if(TARGET OpenCL::OpenCL)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)

    if(libva_FOUND)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
        target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::libva)
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

# ov_ncc_naming_style(FOR_TARGET "${TARGET_NAME}"
#                     SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
#                     ADDITIONAL_INCLUDE_DIRECTORIES
#                         $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>)

#
# Check python snippets
# 

if(ENABLE_PYTHON)
    find_package(PythonInterp REQUIRED)

    file(GLOB_RECURSE PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
    set(PYTHON_SNIPPETS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_snippets")

    if(OV_GENERATOR_MULTI_CONFIG)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python)
    endif()

    foreach(python_snippet IN LISTS PYTHON_SOURCES)
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${PYTHON_SNIPPETS_OUTPUT_DIR}" output_file ${python_snippet})
        add_custom_command(OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${PYTHON_BRIDGE_OUTPUT_DIRECTORY}
                ${PYTHON_EXECUTABLE} ${python_snippet}
            COMMAND ${CMAKE_COMMAND} -E touch ${output_file}
            DEPENDS ${python_snippet} ie_api pyopenvino ov_frontends ov_plugins
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Check python docs snippet ${python_snippet}"
            VERBATIM)
        list(APPEND outputs ${output_file})
    endforeach()

    add_custom_target(${TARGET_NAME}_python ALL DEPENDS ${outputs})
endif()

#
# Example
#

# Detect OpenVINO
find_package(OpenVINO QUIET
             PATHS "${CMAKE_BINARY_DIR}"
             NO_DEFAULT_PATH)
if(NOT OpenVINO_FOUND)
    set(OpenVINO_DIR ${CMAKE_BINARY_DIR})
endif()

set(TARGET_NAME "ov_integration_snippet")
# [cmake:integration_example_cpp]
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)

add_executable(${TARGET_NAME} src/main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# [cmake:integration_example_cpp]

set(TARGET_NAME_C "ov_integration_snippet_c")
# [cmake:integration_example_c]
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenVINO REQUIRED)

add_executable(${TARGET_NAME_C} src/main.c)

target_link_libraries(${TARGET_NAME_C} PRIVATE openvino::runtime::c)

# [cmake:integration_example_c]
