# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.13)
project(python_tools)

if(NOT DEFINED OpenVINO_MAIN_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage QUIET)
endif()
find_package(PythonInterp)

if(PYTHONINTERP_FOUND)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
    message(FATAL_ERROR "Python Interpreter was not found!")
endif()

set(TARGET_NAME "python_tools")

if(WIN32)
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
else()
    set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
endif()

if(ENABLE_PYTHON)

    # creates a copy inside bin directory for developers to have ability running python benchmark_app
    add_custom_target(${TARGET_NAME} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenVINO_MAIN_SOURCE_DIR}/tools/benchmark ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools/benchmark
    )

    ie_cpack_add_component(python_tools_${PYTHON_VERSION})
    ie_cpack_add_component(python_tools)

    install(DIRECTORY ../inference-engine/tools/benchmark_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)

    install(DIRECTORY ../inference-engine/tools/cross_check_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)

    install(FILES README.md
            DESTINATION python/${PYTHON_VERSION}/openvino/tools
            COMPONENT python_tools_${PYTHON_VERSION})

    install(DIRECTORY benchmark/
            DESTINATION python/${PYTHON_VERSION}/openvino/tools
            USE_SOURCE_PERMISSIONS
            COMPONENT python_tools_${PYTHON_VERSION})
    
    ie_cpack(python_tools python_tools_${PYTHON_VERSION})

endif()
