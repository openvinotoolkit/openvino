# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(WHEEL_PACKAGE_NAME "openvino" CACHE STRING "Name of the package")
set(WHEEL_LICENCE_TYPE "OSI Approved :: Apache Software License" CACHE STRING "License type for the package")
set(WHEEL_AUTHOR "Intel Corporation" CACHE STRING "Package author’s name")
set(WHEEL_AUTHOR_EMAIL "openvino_pushbot@intel.com" CACHE STRING "Email address of the package author")
set(WHEEL_DESC "Inference Engine Python* API" CACHE STRING "Short, summary description of the package")
set(WHEEL_VERSION "0.0.0" CACHE STRING "Version of this release")
set(WHEEL_BUILD "000" CACHE STRING "Build number of this release")
set(WHEEL_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE" CACHE STRING "Wheel license file")
set(WHEEL_REQUIREMENTS "${CMAKE_CURRENT_SOURCE_DIR}/meta/openvino.requirements.txt" CACHE STRING "Wheel requirements.txt file")
set(WHEEL_OVERVIEW "${CMAKE_CURRENT_SOURCE_DIR}/meta/pypi_overview.md" CACHE STRING "WHEEL_AUTHOR detailed description")

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_ENV   "${CMAKE_CURRENT_SOURCE_DIR}/.env.in")

set(CORE_LIBS_DIR ${IE_CPACK_RUNTIME_PATH})
set(PLUGINS_LIBS_DIR ${IE_CPACK_RUNTIME_PATH})
set(NGRAPH_LIBS_DIR ${IE_CPACK_IE_DIR}/../ngraph/lib)
set(PY_PACKAGES_DIR ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION})
set(TBB_LIBS_DIR deployment_tools/inference_engine/external/tbb/lib)


if(APPLE)
    set(WHEEL_PLATFORM macosx_10_15_x86_64)
elseif(UNIX)
    set(WHEEL_PLATFORM manylinux2014_x86_64)
elseif(WIN32)
    set(WHEEL_PLATFORM win_amd64)
    set(TBB_LIBS_DIR deployment_tools/inference_engine/external/tbb/bin)
else()
    message(FATAL_ERROR "This platform is not supported")
endif()

configure_file(${SETUP_ENV} "${CMAKE_CURRENT_SOURCE_DIR}/.env")

add_custom_target(ie_wheel ALL DEPENDS ie_libraries ie_plugins ie_api _pyngraph)

add_custom_command(TARGET ie_wheel
    POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} bdist_wheel
        --dist-dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/wheels
        --build=${WHEEL_BUILD}
        --plat-name=${WHEEL_PLATFORM}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${WHEEL_PACKAGE_NAME}"
    VERBATIM
)
