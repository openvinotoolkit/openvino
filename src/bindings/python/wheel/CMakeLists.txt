# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Define proper package name
#

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')"
                OUTPUT_VARIABLE PYTHON_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')"
                OUTPUT_VARIABLE ABI_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags.platform_tags())}')"
                OUTPUT_VARIABLE PLATFORM_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)

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
        set(_arch "armvl7")
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

    string(REPLACE "." "_" _ov_glibc_version "${OV_GLIBC_VERSION}")
    set(manylinux "manylinux_${_ov_glibc_version}")

    # convert to well-known formats according to PEP 600
    if(manylinux STREQUAL "manylinux_2_5")
        set(manylinux "manylinux1")
    elseif(manylinux STREQUAL "manylinux_2_12")
        set(manylinux "manylinux2010")
    elseif(manylinux STREQUAL "manylinux_2_17")
        set(manylinux "manylinux2014")
    endif()

    set(PLATFORM_TAG "${manylinux}_${_arch}")
endif()

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

#
# create target for openvino.wheel
#

execute_process(COMMAND ${Python3_EXECUTABLE} -m pip --version
                OUTPUT_VARIABLE pip_version OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX MATCH "pip[ ]+([\\.0-9]*)" pip_version "${pip_version}")
set(pip_version ${CMAKE_MATCH_1})

if(pip_version VERSION_GREATER_EQUAL 22.0)
    set(wheel_build_command
        ${Python3_EXECUTABLE} -m pip wheel
            --no-deps
            --wheel-dir ${openvino_wheels_output_dir}
            # --verbose
            --build-option --build-number=${WHEEL_BUILD}
            --build-option --plat-name=${PLATFORM_TAG}
            "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    set(wheel_build_command
        ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
            --quiet
            --no-user-cfg
            bdist_wheel
            --dist-dir ${openvino_wheels_output_dir}
            --build-number=${WHEEL_BUILD}
            --plat-name=${PLATFORM_TAG})
endif()

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${setup_py_env}
        ${wheel_build_command}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/build_${pyversion}"
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

# install

ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_WHEELS} HIDDEN)

install(FILES ${openvino_wheel_path}
        DESTINATION ${OV_CPACK_WHEELSDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_WHEELS}
        ${OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL})
