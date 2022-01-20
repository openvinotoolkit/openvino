# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINOPython DESCRIPTION "OpenVINO Runtime Python bindings")

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)

set(LIBRARY_OUTPUT_DIRECTORY_BIN ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_subdirectory(src)

if(ENABLE_WHEEL)
    add_subdirectory(wheel)
endif()

if(ENABLE_OV_CORE_UNIT_TESTS)
    add_subdirectory(tests/mock/ov_mock_py_frontend)
    add_dependencies(pyopenvino ov_mock_py_frontend)
    set_target_properties(ov_mock_py_frontend PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            COMPILE_PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN}
            PDB_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY_BIN})

    add_subdirectory(tests/mock/pyngraph_fe_mock_api)
    add_dependencies(pyopenvino pybind_mock_frontend)
endif()

if(InferenceEngineDeveloperPackage_FOUND)
    ie_cpack(${IE_CPACK_COMPONENTS_ALL})
endif()
