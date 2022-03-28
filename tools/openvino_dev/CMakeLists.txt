# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(WHEEL_VERSION "${OpenVINO_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)
set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")

set(openvino_wheel_name "openvino_dev-${WHEEL_VERSION}-${WHEEL_BUILD}-py3-none-any.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env OPENVINO_VERSION=${WHEEL_VERSION}
    ${PYTHON_EXECUTABLE} ${SETUP_PY} clean bdist_wheel
        --dist-dir ${openvino_wheels_output_dir}
        --build=${WHEEL_BUILD}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(openvino_dev_wheel ALL DEPENDS ${openvino_wheel_path})

# install

ie_cpack_add_component(python_wheels)

install(FILES ${openvino_wheel_path}
        DESTINATION tools
        COMPONENT python_wheels)
