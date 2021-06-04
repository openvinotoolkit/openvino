# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)
project(python_tools)

if(NOT DEFINED OpenVINO_MAIN_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage QUIET)
endif()

if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(Python3_VERSION python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpreter was not found!")
endif()

set(TARGET_NAME "python_tools")

if(WIN32)
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${Python3_VERSION}/openvino)
else()
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${Python3_VERSION}/openvino)
endif()

if(ENABLE_PYTHON)
    # creates a copy inside bin directory for developers to have ability running python benchmark_app
    add_custom_target(${TARGET_NAME} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenVINO_MAIN_SOURCE_DIR}/tools/benchmark ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools/benchmark
    )

    ie_cpack_add_component(python_tools_${Python3_VERSION})
    ie_cpack_add_component(python_tools)

    install(DIRECTORY ../inference-engine/tools/benchmark_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)

    install(DIRECTORY ../inference-engine/tools/cross_check_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)

    install(FILES README.md
            DESTINATION python/${Python3_VERSION}/openvino/tools
            COMPONENT python_tools_${Python3_VERSION})

    install(DIRECTORY benchmark
            DESTINATION python/${Python3_VERSION}/openvino/tools
            USE_SOURCE_PERMISSIONS
            COMPONENT python_tools_${Python3_VERSION})
    
    ie_cpack(python_tools python_tools_${Python3_VERSION})
endif()
