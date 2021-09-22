# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(WHEEL_PACKAGE_NAME "openvino" CACHE STRING "Name of the package")
set(WHEEL_LICENCE_TYPE "OSI Approved :: Apache Software License" CACHE STRING "License type for the package")
set(WHEEL_AUTHOR "Intel Corporation" CACHE STRING "Package author’s name")
set(WHEEL_AUTHOR_EMAIL "openvino_pushbot@intel.com" CACHE STRING "Email address of the package author")
set(WHEEL_DESC "Inference Engine Python* API" CACHE STRING "Short, summary description of the package")
set(WHEEL_URL "https://docs.openvinotoolkit.org/latest/index.html" CACHE STRING "Home page url")
set(WHEEL_DOWNLOAD_URL "https://github.com/openvinotoolkit/openvino/tags" CACHE STRING "Download page url")
set(WHEEL_VERSION "0.0.0" CACHE STRING "Version of this release")
set(WHEEL_BUILD "000" CACHE STRING "Build number of this release")
set(WHEEL_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE" CACHE STRING "Wheel license file")
set(WHEEL_REQUIREMENTS "${CMAKE_CURRENT_SOURCE_DIR}/meta/openvino.requirements.txt" CACHE STRING "Wheel requirements.txt file")
set(WHEEL_OVERVIEW "${CMAKE_CURRENT_SOURCE_DIR}/meta/pypi_overview.md" CACHE STRING "Detailed description")

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_ENV "${CMAKE_CURRENT_SOURCE_DIR}/.env.in")

set(PY_PACKAGES_DIR ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION})
set(TBB_LIBS_DIR runtime/3rdparty/tbb/lib)

if(APPLE)
    set(WHEEL_PLATFORM macosx_10_15_x86_64)
elseif(UNIX)
    set(WHEEL_PLATFORM manylinux2014_x86_64)
elseif(WIN32)
    set(WHEEL_PLATFORM win_amd64)
    set(TBB_LIBS_DIR runtime/3rdparty/tbb/bin)
else()
    message(FATAL_ERROR "This platform is not supported")
endif()

configure_file(${SETUP_ENV} "${CMAKE_CURRENT_SOURCE_DIR}/.env" @ONLY)

if(LINUX)
    find_host_program(patchelf_program
                      NAMES patchelf
                      DOC "Path to patchelf tool")
    if(NOT patchelf_program)
        message(FATAL_ERROR "patchelf is not found, which is needed to build ie_wheel")
    endif()
endif()

# create target for openvino.wheel

set(openvino_wheel_deps ie_api offline_transformations_api)
foreach(_target ie_libraries ie_plugins _pyngraph)
    if(TARGET ${_target})
        list(APPEND openvino_wheel_deps ${_target})
    endif()
endforeach()

set(cp_python "cp${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}")
set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${cp_python}-${cp_python}-${WHEEL_PLATFORM}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} clean bdist_wheel
        --dist-dir ${openvino_wheels_output_dir}
        --build=${WHEEL_BUILD}
        --plat-name=${WHEEL_PLATFORM}
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_CURRENT_SOURCE_DIR}/.env"
    DEPENDS ${openvino_wheel_deps}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel DEPENDS ${openvino_wheel_path})

set_target_properties(ie_wheel
    PROPERTIES ADDITIONAL_CLEAN_FILES "${openvino_wheel_path}")
