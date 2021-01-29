# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(WHEEL_PACKAGE_NAME "openvino" CACHE STRING "Name of the package")
set(WHEEL_LICENCE_TYPE "OSI Approved :: Apache Software License" CACHE STRING "License for the package")
set(WHEEL_AUTHOR "Intel Corporation" CACHE STRING "Package author’s name")
set(WHEEL_AUTHOR_EMAIL "openvino_pushbot@intel.com" CACHE STRING "Email address of the package author")
set(WHEEL_DESC "Inference Engine Python* API" CACHE STRING "Short, summary description of the package")
set(WHEEL_VERSION "0.0.0" CACHE STRING "Version of this release")
set(WHEEL_BUILD "000" CACHE STRING "Build number of this release")
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OVERVIEW_IN "${CMAKE_CURRENT_SOURCE_DIR}/meta/pypi_overview.md")
set(LICENSE_IN "${CMAKE_SOURCE_DIR}/LICENSE")
set(REQUIREMENTS_IN "${CMAKE_CURRENT_SOURCE_DIR}/meta/openvino.requirements.txt")


if(APPLE)
    set(WHEEL_PLATFORM macosx_10_15_x86_64)
elseif(UNIX)
    set(WHEEL_PLATFORM manylinux2014_x86_64)
elseif(WIN32)
    set(WHEEL_PLATFORM win_amd64)
else()
    message(FATAL_ERROR "This platform is not supported")
endif()

# set version using CI value
if (WHEEL_VERSION STREQUAL  "0.0.0")
    string(REGEX MATCH "([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)" WHEEL_VERSION "${CI_BUILD_NUMBER}")
endif()

# prepare setup.py
configure_file("${SETUP_PY_IN}" "${SETUP_PY}")
configure_file("${OVERVIEW_IN}" "${CMAKE_CURRENT_BINARY_DIR}/pypi_overview.md" COPYONLY)
configure_file("${REQUIREMENTS_IN}" "${CMAKE_CURRENT_BINARY_DIR}/requirements.txt" COPYONLY)
configure_file("${LICENSE_IN}" "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt" COPYONLY)

add_custom_target(ie_wheel ALL DEPENDS ie_libraries ie_plugins ie_api _pyngraph)

add_custom_command(TARGET ie_wheel
    POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} clean bdist_wheel
        --dist-dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/wheels
        --build=${WHEEL_BUILD}
        --dist-dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/wheels
        --plat-name=${WHEEL_PLATFORM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${WHEEL_PACKAGE_NAME}"
    VERBATIM
)
