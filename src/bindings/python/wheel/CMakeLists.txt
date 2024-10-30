# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Define proper package name
#

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')"
                OUTPUT_VARIABLE PYTHON_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PYTHON_TAG)
    message(FATAL_ERROR "Failed to detect Python Tag via wheel.vendored.packaging.tags. Please, check 'wheel' dependency version update")
endif()

execute_process(COMMAND ${Python3_EXECUTABLE} -c "from setuptools.command.bdist_wheel import get_abi_tag; print(f'{get_abi_tag()}')"
                OUTPUT_VARIABLE ABI_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ABI_TAG)
    message(FATAL_ERROR "Failed to detect ABI Tag via setuptools.command.bdist_wheel. Please, check 'setuptools' dependency version update")
endif()

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags.platform_tags())}')"
                OUTPUT_VARIABLE PLATFORM_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT PLATFORM_TAG)
    message(FATAL_ERROR "Failed to detect Platform Tag via wheel.vendored.packaging.tags. Please, check 'wheel' dependency version update")
endif()

# defines wheel architecture part of `PLATFORM_TAG`
macro(_ov_platform_arch)
    if(AARCH64)
        if(APPLE)
            set(_arch "arm64")
        else()
            set(_arch "aarch64")
        endif()
    elseif(UNIVERSAL2)
        set(_arch "universal2")
    elseif(ARM)
        set(_arch "armv7l")
    elseif(X86_64)
        set(_arch "x86_64")
    elseif(X86)
        set(_arch "i686")
    elseif(RISCV64)
        set(_arch "riscv64")
    else()
        message(FATAL_ERROR "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endmacro()

# For macOS and Linux `PLATFORM_TAG` is not always correctly detected
# So, we need to add our-own post-processing
if(APPLE)
    _ov_platform_arch()

    if(CMAKE_OSX_DEPLOYMENT_TARGET)
        set(_macos_target_version "${CMAKE_OSX_DEPLOYMENT_TARGET}")
        if(_macos_target_version MATCHES "^1[0-9]$")
            set(_macos_target_version "${CMAKE_OSX_DEPLOYMENT_TARGET}.0")
        endif()
        string(REPLACE "." "_" _macos_target_version "${_macos_target_version}")
    else()
        string(REGEX MATCH "1[0-9]_[0-9]+" _macos_target_version ${PLATFORM_TAG})
    endif()

    # common platform tag looks like macosx_<macos major>_<macos minor>_<arch>
    if(_arch AND _macos_target_version)
        set(PLATFORM_TAG "macosx_${_macos_target_version}_${_arch}")
    endif()
elseif(LINUX)
    _ov_platform_arch()

    if(OPENVINO_GNU_LIBC)
        set(platform "manylinux")
    elseif(OPENVINO_MUSL_LIBC)
        set(platform "musllinux")
    else()
        message(FATAL_ERROR "Undefined libc type")
    endif()

    string(REPLACE "." "_" _ov_libc_version "${OV_LIBC_VERSION}")
    set(platform "${platform}_${_ov_libc_version}")

    # convert to well-known formats according to PEP 600
    if(platform STREQUAL "manylinux_2_5")
        set(platform "manylinux1")
    elseif(platform STREQUAL "manylinux_2_12")
        set(platform "manylinux2010")
    elseif(platform STREQUAL "manylinux_2_17")
        set(platform "manylinux2014")
    endif()

    set(PLATFORM_TAG "${platform}_${_arch}")
endif()

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

#
# create target for openvino.wheel
#

# for --config-setting explanation see https://github.com/pypa/setuptools/issues/2491
set(wheel_build_command
    ${Python3_EXECUTABLE} -m build "${CMAKE_CURRENT_SOURCE_DIR}"
        --outdir ${openvino_wheels_output_dir}
        --config-setting=--build-option=--build-number=${WHEEL_BUILD}
        --config-setting=--build-option=--plat-name=${PLATFORM_TAG}
        --config-setting=--quiet
        --wheel)

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${setup_py_env} ${wheel_build_command}
    DEPENDS ${ov_setup_py_deps}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

set(fdupes_report ${CMAKE_CURRENT_BINARY_DIR}/fdupes_report.txt)
add_custom_command(OUTPUT "${fdupes_report}"
    COMMAND ${CMAKE_COMMAND}
        -D Python3_EXECUTABLE=${Python3_EXECUTABLE}
        -D WORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        -D WHEEL_VERSION=${WHEEL_VERSION}
        -D PACKAGE_FILE=${openvino_wheel_path}
        -D REPORT_FILE=${fdupes_report}
        -D CMAKE_SHARED_LIBRARY_SUFFIX=${CMAKE_SHARED_LIBRARY_SUFFIX}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/fdupes_check.cmake"
    DEPENDS "${openvino_wheel_path}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Run 'fdupes' checks for wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel ALL DEPENDS ${openvino_wheel_path} ${fdupes_report})

add_custom_command(
    TARGET ie_wheel
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/build_${pyversion}" || ${CMAKE_COMMAND} -E true # w/a this ensures the command always succeeds
    DEPENDS ${openvino_wheel_path}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Cleaning up Python wheel build for wheel ${openvino_wheel_name}"
    VERBATIM)

# install

ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_WHEELS} HIDDEN)

install(FILES ${openvino_wheel_path}
        DESTINATION ${OV_CPACK_WHEELSDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_WHEELS}
        ${OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL})
