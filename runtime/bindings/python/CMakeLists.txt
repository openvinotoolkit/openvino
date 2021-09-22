# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINOPython DESCRIPTION "OpenVINO Runtime Python bindings")

add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)

find_package(PythonLibs 3 REQUIRED)

if(PYTHONLIBS_VERSION_STRING MATCHES "^([0-9]+)\.([0-9]+).*")
    set(PYTHON_VERSION python${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
else()
    message(FATAL_ERROR "Failed to extract python major.minor from ${PYTHONLIBS_VERSION_STRING}")
endif()

if(ENABLE_CONDA_FOLDER)
    set(PYTHON_COMPONENT conda_${PYTHON_VERSION})
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/Conda/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/Conda/${PYTHON_VERSION}/openvino)
    endif()
else()
    set(PYTHON_COMPONENT ${PYTHON_VERSION})
    if(WIN32)
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<CONFIG>/python_api/${PYTHON_VERSION}/openvino)
    else()
        set(PYTHON_BRIDGE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_api/${PYTHON_VERSION}/openvino)
    endif()
endif()

message("AUTPUUUUUUUUUT  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

set(LIBRARY_OUTPUT_DIRECTORY_BIN ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_subdirectory(src)

if(NGRAPH_UNIT_TEST_ENABLE)
    add_subdirectory(tests/mock/mock_py_ngraph_frontend)
    add_dependencies(_pyngraph mock_py_ngraph_frontend)
    set_target_properties(mock_py_ngraph_frontend PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            COMPILE_PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN})

    add_subdirectory(tests/mock/pyngraph_fe_mock_api)
    add_dependencies(_pyngraph pybind_mock_frontend)
endif()
