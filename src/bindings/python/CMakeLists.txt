# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINOPython DESCRIPTION "OpenVINO Runtime Python bindings")

#
# Packages & settings
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

set(PYTHON_SOURCE_DIR ${OpenVINOPython_SOURCE_DIR}/src)


#
# Check python requirements
#

set(ov_python_req "${OpenVINOPython_SOURCE_DIR}/requirements.txt")
set(ie_python_req "cython>=0.29.22")

function(ov_check_python_build_conditions)
    # user explicitly specified ENABLE_PYTHON=ON
    if(ENABLE_PYTHON)
        set(find_package_mode REQUIRED)
        set(message_mode FATAL_ERROR)
    else()
        set(find_package_mode QUIET)
        set(message_mode WARNING)
    endif()

    # Try to find python3 and its libs
    find_host_package(PythonInterp 3 ${find_package_mode})
    if(PYTHONINTERP_FOUND)
        function(_ov_find_python_libs_new)
            set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION_STRING})
            list(APPEND CMAKE_MODULE_PATH "${OpenVINOPython_SOURCE_DIR}/thirdparty/pybind11/tools")
            # use libraries with the same version as python itself
            find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} EXACT ${find_package_mode})
            set(PYTHONLIBSNEW_FOUND ${PYTHONLIBS_FOUND} PARENT_SCOPE)
        endfunction()
        # try to find python libraries
        _ov_find_python_libs_new()
        if(PYTHONLIBSNEW_FOUND)
            find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT ${find_package_mode})
        endif()
        if(NOT PYTHONLIBS_FOUND)
            message(${message_mode} "Python development libraries are not found. OpenVINO Python API will be turned off (ENABLE_PYTHON is OFF)")
        endif()
    else()
        message(${message_mode} "Python 3.x interpreter is not found. OpenVINO Python API will be turned off (ENABLE_PYTHON is OFF)")
    endif()

    # check pyopenvino requirements to OV 2.0 API
    ov_check_pip_packages(REQUIREMENTS_FILE ${ov_python_req}
                          RESULT_VAR ov_python_req_FOUND
                          WARNING_MESSAGE "install python3 -m install -r ${ov_python_req} for OV API 2.0 requirements"
                          MESSAGE_MODE TRACE)
    # ov_python_req are not mandatory for build
    set(ov_python_req_FOUND ON)

    # check for Cython requirement for build IE API 1.0
    ov_check_pip_package(REQUIREMENT ${ie_python_req}
                         RESULT_VAR ie_python_req_FOUND
                         WARNING_MESSAGE "install python3 -m install ${ie_python_req} for IE API 1.0 requirements"
                         MESSAGE_MODE TRACE)

    # cython can be installed as a debian package, so pip requirements can be unsatisfied
    # so, let's check to find cython anyway
    if(NOT ie_python_req_FOUND)
        find_package(Cython QUIET
                     PATHS "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/cmake"
                     NO_CMAKE_FIND_ROOT_PATH
                     NO_DEFAULT_PATH)
        if(CYTHON_VERSION VERSION_GREATER_EQUAL 0.29)
            set(ie_python_req_FOUND ON)
        else()
            message(${message_mode} "Python module '${ie_python_req}' is missed, IE Python API 1.0 will not be built (ENABLE_PYTHON is OFF)")
        endif()
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_DEBUG_POSTFIX)
        set(python_debug ON)
        message(${message_mode} "Building python bindings in debug configuration is not supported on your platform (ENABLE_PYTHON is OFF)")
    else()
        set(python_debug OFF)
    endif()

    if(PYTHONLIBS_FOUND AND ov_python_req_FOUND AND ie_python_req_FOUND AND NOT python_debug)
        set(ENABLE_PYTHON_DEFAULT ON PARENT_SCOPE)
    else()
        set(ENABLE_PYTHON_DEFAULT OFF PARENT_SCOPE)
    endif()

    # to disable API 1.0
    set(ie_python_req_FOUND ${ie_python_req_FOUND} PARENT_SCOPE)
endfunction()

ov_check_python_build_conditions()

ie_option(ENABLE_PYTHON "Enables OpenVINO Python API build" ${ENABLE_PYTHON_DEFAULT})

if(NOT ENABLE_PYTHON)
    if(CMAKE_SOURCE_DIR STREQUAL OpenVINOPython_SOURCE_DIR)
        message(FATAL_ERROR "Python OpenVINO API requirements are not satisfied. Please, install ${ie_python_req} and ${ov_python_req}")
    else()
        return()
    endif()
endif()

#
# Check for wheel package
#

# user explicitly specified ENABLE_WHEEL=ON
if(ENABLE_WHEEL)
    set(find_package_mode REQUIRED)
    set(message_mode FATAL_ERROR)
else()
    set(find_package_mode QUIET)
    set(message_mode WARNING)
endif()

set(wheel_reqs "${OpenVINOPython_SOURCE_DIR}/wheel/requirements-dev.txt")
ov_check_pip_packages(REQUIREMENTS_FILE "${OpenVINOPython_SOURCE_DIR}/wheel/requirements-dev.txt"
                      RESULT_VAR wheel_req_FOUND
                      MESSAGE_MODE WARNING)

if(wheel_req_FOUND AND "${CMAKE_SOURCE_DIR}" STREQUAL "${OpenVINO_SOURCE_DIR}")
    set(ENABLE_WHEEL_DEFAULT ON)
else()
    set(ENABLE_WHEEL_DEFAULT OFF)
endif()

if(LINUX)
    find_host_program(patchelf_program
                      NAMES patchelf
                      DOC "Path to patchelf tool")
    if(NOT patchelf_program)
        set(ENABLE_WHEEL_DEFAULT OFF)
        message(${message_mode} "patchelf is not found. It is required to build OpenVINO Runtime wheel. Install via apt-get install patchelf")
    endif()
endif()

# this option should not be a part of InferenceEngineDeveloperPackage
# since wheels can be built only together with main OV build
ie_dependent_option(ENABLE_WHEEL "Build wheel packages for PyPI" ${ENABLE_WHEEL_DEFAULT} "ENABLE_PYTHON" OFF)

#
# Build the code
#

if(DISABLE_PYBIND11)
    message(WARNING "NOTE: Python API for OpenVINO 2.0 is disabled")
else()
    add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)
    add_subdirectory(src/compatibility/pyngraph)
    add_subdirectory(src/pyopenvino)
endif()

if(ie_python_req_FOUND)
    add_subdirectory(src/compatibility/openvino)
else()
    message(WARNING "NOTE: Python API for OpenVINO 1.0 is disabled")
endif()

if(ENABLE_WHEEL)
    add_subdirectory(wheel)
endif()

if(ENABLE_OV_CORE_UNIT_TESTS AND NOT DISABLE_PYBIND11)
    set(LIBRARY_OUTPUT_DIRECTORY_BIN ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    add_subdirectory(tests/mock/mock_py_frontend)
    add_dependencies(pyopenvino openvino_mock_py_frontend)
    set_target_properties(openvino_mock_py_frontend PROPERTIES
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
