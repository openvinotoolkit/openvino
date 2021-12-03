# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# C++ tools
#

add_subdirectory(compile_tool)

#
# Python tools
#

# install deployment_manager

ie_cpack_add_component(deployment_manager REQUIRED)

install(DIRECTORY deployment_manager
        DESTINATION tools
        COMPONENT deployment_manager
        USE_SOURCE_PERMISSIONS)

# copy benchmark_app scripts

if(ENABLE_PYTHON)
    find_package(PythonInterp 3 REQUIRED)
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})

    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
    endif()

    # creates a copy inside bin directory for developers to have ability running python benchmark_app
    add_custom_target(python_tools_benchmark ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tool/openvino/tools/benchmark
                                                   ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools/benchmark
    )

    # install BA tool for tests
    install(FILES benchmark_tool/README.md
                  benchmark_tool/requirements.txt
            DESTINATION tools/benchmark_tool
            COMPONENT tests
            EXCLUDE_FROM_ALL)

    install(PROGRAMS benchmark_tool/benchmark_app.py
            DESTINATION tools/benchmark_tool
            COMPONENT tests
            EXCLUDE_FROM_ALL)

    # install openvino/tools/benchmark as a python package
    install(DIRECTORY benchmark_tool/openvino/tools/benchmark
            DESTINATION python/${PYTHON_VERSION}/openvino/tools
            USE_SOURCE_PERMISSIONS
            COMPONENT tests
            EXCLUDE_FROM_ALL)
endif()

# build and install openvino-dev wheel

if(ENABLE_WHEEL)
    add_subdirectory(openvino_dev)
endif()
