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
set(WHEEL_VERSION "${IE_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${IE_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)
set(WHEEL_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE" CACHE STRING "Wheel license file")
set(WHEEL_REQUIREMENTS "${CMAKE_CURRENT_SOURCE_DIR}/meta/openvino.requirements.txt" CACHE STRING "Wheel requirements.txt file")
set(WHEEL_OVERVIEW "${CMAKE_CURRENT_SOURCE_DIR}/meta/pypi_overview.md" CACHE STRING "Detailed description")

set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_ENV "${CMAKE_CURRENT_SOURCE_DIR}/.env.in")
set(SETUP_ENV_OUT "${CMAKE_CURRENT_SOURCE_DIR}/.env")

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

configure_file(${SETUP_ENV} ${SETUP_ENV_OUT} @ONLY)

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
foreach(_target ov_runtime_libraries ie_plugins _pyngraph pyopenvino)
    if(TARGET ${_target})
        list(APPEND openvino_wheel_deps ${_target})
    endif()
endforeach()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')" OUTPUT_VARIABLE PYTHON_ABI)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')" OUTPUT_VARIABLE INTERPRETER)
string(STRIP ${PYTHON_ABI} PYTHON_ABI)
string(STRIP ${INTERPRETER} INTERPRETER)

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${INTERPRETER}-${PYTHON_ABI}-${WHEEL_PLATFORM}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} clean bdist_wheel
        --dist-dir ${openvino_wheels_output_dir}
        --build=${WHEEL_BUILD}
        --plat-name=${WHEEL_PLATFORM}
    # COMMAND ${CMAKE_COMMAND} -E remove ${SETUP_ENV_OUT}
    DEPENDS ${openvino_wheel_deps} ${SETUP_ENV_OUT}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel ALL DEPENDS ${openvino_wheel_path})

# install

ie_cpack_add_component(python_wheels)

install(FILES ${openvino_wheel_path}
        DESTINATION tools
        COMPONENT python_wheels)
