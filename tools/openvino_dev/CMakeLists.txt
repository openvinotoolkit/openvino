# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required (VERSION 3.13)
project(openvino_dev)

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()


set(WHEEL_VERSION "${IE_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${IE_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)
set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

set(openvino_wheel_name "openvino_dev-${WHEEL_VERSION}-${WHEEL_BUILD}-py3-none-any.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E env OPENVINO_VERSION=${WHEEL_VERSION}
    ${PYTHON_EXECUTABLE} ${SETUP_PY} clean bdist_wheel
        --dist-dir ${openvino_wheels_output_dir}
        --build=${WHEEL_BUILD}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(openvino_dev_wheel ALL DEPENDS ${openvino_wheel_path})

# outbound requirements.txt files for openvino-dev package
set(REQUIREMENTS_IN "${CMAKE_CURRENT_SOURCE_DIR}/requirements_dev.in")
set(EXTRAS_LIST _ caffe caffe2 kaldi mxnet onnx pytorch tensorflow tensorflow2)

foreach(EXTRAS ${EXTRAS_LIST})
    if(${EXTRAS} STREQUAL "_")
        set(REQUIREMENTS_OUT "requirements.txt")
        set(EXTRAS "")
    else()
        set(REQUIREMENTS_OUT "requirements_${EXTRAS}.txt")
        set(EXTRAS "[${EXTRAS}]")
    endif()
    configure_file(${REQUIREMENTS_IN} ${REQUIREMENTS_OUT})

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${REQUIREMENTS_OUT}
        DESTINATION tools
        COMPONENT openvino_dev_req_files
        EXCLUDE_FROM_ALL)
endforeach()

if(InferenceEngineDeveloperPackage_FOUND)
    ie_cpack_add_component(openvino_dev_req_files)
    ie_cpack(${IE_CPACK_COMPONENTS_ALL})
endif()


