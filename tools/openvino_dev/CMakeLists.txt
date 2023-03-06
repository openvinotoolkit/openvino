# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINODevPython DESCRIPTION "OpenVINO Python Development tools")

#
# Packages & settings
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    get_filename_component(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)
endif()

if(NOT IEDevScripts_FOUND)
    find_package(IEDevScripts REQUIRED
                PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
                NO_CMAKE_FIND_ROOT_PATH
                NO_DEFAULT_PATH)
endif()

# define version (syncronize with src/bindings/python/wheel/CMakeLists.txt)

if(DEFINED ENV{CI_BUILD_DEV_TAG} AND NOT "$ENV{CI_BUILD_DEV_TAG}" STREQUAL "")
    set(WHEEL_VERSION "${OpenVINO_VERSION}.$ENV{CI_BUILD_DEV_TAG}" CACHE STRING "Version of this release" FORCE)
    set(wheel_pre_release ON)
else()
    set(WHEEL_VERSION ${OpenVINO_VERSION} CACHE STRING "Version of this release" FORCE)
endif()
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)

# outbound requirements.txt files for openvino-dev package

ie_cpack_add_component(${OV_CPACK_COMP_DEV_REQ_FILES} HIDDEN)

set(REQUIREMENTS_IN "${CMAKE_CURRENT_SOURCE_DIR}/requirements_dev.txt.in")
set(EXTRAS_LIST _ caffe kaldi mxnet onnx pytorch tensorflow tensorflow2)

foreach(EXTRAS IN LISTS EXTRAS_LIST)
    if(EXTRAS STREQUAL "_")
        set(REQUIREMENTS_OUT "requirements.txt")
        set(EXTRAS "")
    else()
        set(REQUIREMENTS_OUT "requirements_${EXTRAS}.txt")
        set(EXTRAS "[${EXTRAS}]")
    endif()
    configure_file(${REQUIREMENTS_IN} ${REQUIREMENTS_OUT})

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${REQUIREMENTS_OUT}
            DESTINATION ${OV_CPACK_DEVREQDIR}
            COMPONENT ${OV_CPACK_COMP_DEV_REQ_FILES}
            EXCLUDE_FROM_ALL)
endforeach()

# openvino_dev build

if(NOT ENABLE_WHEEL)
    return()
endif()

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

set(openvino_wheel_name "openvino_dev-${WHEEL_VERSION}-${WHEEL_BUILD}-py3-none-any.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy "${OpenVINO_SOURCE_DIR}/thirdparty/open_model_zoo/licensing/omz-third-party-programs.txt" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${OpenVINO_SOURCE_DIR}/licensing/dev-third-party-programs.txt" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${OpenVINO_SOURCE_DIR}/LICENSE" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/readme.txt" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env OPENVINO_VERSION=${WHEEL_VERSION}
        ${PYTHON_EXECUTABLE} ${SETUP_PY} bdist_wheel
            --dist-dir ${openvino_wheels_output_dir}
            --build=${WHEEL_BUILD}
            --quiet
    COMMAND ${CMAKE_COMMAND} -E env OPENVINO_VERSION=${WHEEL_VERSION}
        ${PYTHON_EXECUTABLE} ${SETUP_PY} clean
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(openvino_dev_wheel ALL DEPENDS ${openvino_wheel_path})

if(TARGET ie_wheel)
    add_dependencies(openvino_dev_wheel ie_wheel)
endif()

# install

ie_cpack_add_component(${OV_CPACK_COMP_PYTHON_WHEELS} HIDDEN)

install(FILES ${openvino_wheel_path}
        DESTINATION ${OV_CPACK_WHEELSDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_WHEELS})
