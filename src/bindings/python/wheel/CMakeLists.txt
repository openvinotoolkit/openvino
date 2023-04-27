# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# define version (syncronize with tools/openvino_dev/CMakeLists.txt)

if(DEFINED ENV{CI_BUILD_DEV_TAG} AND NOT "$ENV{CI_BUILD_DEV_TAG}" STREQUAL "")
    set(WHEEL_VERSION "${OpenVINO_VERSION}.$ENV{CI_BUILD_DEV_TAG}" CACHE STRING "Version of this release" FORCE)
    set(wheel_pre_release ON)
else()
    set(WHEEL_VERSION ${OpenVINO_VERSION} CACHE STRING "Version of this release" FORCE)
endif()
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)


#
# Common vars used by setup.py
#

set(PY_PACKAGES_DIR ${OV_CPACK_PYTHONDIR})
set(TBB_LIBS_DIR runtime/3rdparty/tbb/lib)
if(WIN32)
    set(TBB_LIBS_DIR runtime/3rdparty/tbb/bin)
endif()
set(PUGIXML_LIBS_DIR runtime/3rdparty/pugixml/lib)

#
# Dependencies
#

foreach(_target ie_api constants _pyngraph openvino_c pyopenvino ov_plugins ov_frontends py_ov_frontends)
    if(TARGET ${_target})
        list(APPEND openvino_wheel_deps ${_target})
    endif()
endforeach()

file(GLOB_RECURSE compat_ngraph_py_files ${OpenVINOPython_SOURCE_DIR}/src/compatibility/*.py)
file(GLOB_RECURSE openvino_py_files ${OpenVINOPython_SOURCE_DIR}/src/openvino/*.py)

list(APPEND openvino_wheel_deps
    ${openvino_py_files}
    ${compat_ngraph_py_files}
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    "${OpenVINOPython_SOURCE_DIR}/requirements.txt"
    "${OpenVINOPython_SOURCE_DIR}/wheel/readme.txt"
    "${OpenVINO_SOURCE_DIR}/LICENSE"
    "${OpenVINO_SOURCE_DIR}/licensing/onednn_third-party-programs.txt"
    "${OpenVINO_SOURCE_DIR}/licensing/runtime-third-party-programs.txt"
    "${OpenVINO_SOURCE_DIR}/licensing/tbb_third-party-programs.txt"
    "${OpenVINO_SOURCE_DIR}/docs/install_guides/pypi-openvino-rt.md")

if(wheel_pre_release)
    list(APPEND openvino_wheel_deps
        "${OpenVINO_SOURCE_DIR}/docs/install_guides/pre-release-note.md")
endif()

#
# Define proper package name
#

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')"
                OUTPUT_VARIABLE PYTHON_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')"
                OUTPUT_VARIABLE ABI_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags.platform_tags())}')"
                OUTPUT_VARIABLE PLATFORM_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)

# defines wheel architecture part of `PLATFORM_TAG`
macro(_ov_platform_arch)
    if(AARCH64)
        if(APPLE)
            set(_arch "arm64")
        else()
            set(_arch "aarch64")
        endif()
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
if(APPLE AND DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    _ov_platform_arch()

    set(_macos_min_version "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(_macos_min_version MATCHES "^1[0-9]$")
        set(_macos_min_version "${CMAKE_OSX_DEPLOYMENT_TARGET}.0")
    endif()
    string(REPLACE "." "_" _macos_min_version "${_macos_min_version}")

    # common platform tag looks like macosx_<macos major>_<macos minor>_<arch>
    if(_arch AND _macos_min_version)
        set(PLATFORM_TAG "macosx_${_macos_min_version}_${_arch}")
    endif()
elseif(LINUX)
    _ov_platform_arch()

    if(CMAKE_CROSSCOMPILING)
        # TODO: think which proper tag is needed for arm / aarch64
        set(PLATFORM_TAG "linux_${_arch}")
    else()
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
endif()

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

#
# create target for openvino.wheel
#

set(wheel_env ${CMAKE_COMMAND} -E env
    WHEEL_VERSION=${WHEEL_VERSION}
    WHEEL_BUILD=${WHEEL_BUILD}
    OPENVINO_BUILD_DIR=${OpenVINO_BINARY_DIR}
    OPENVINO_PYTHON_BUILD_DIR=${OpenVINOPython_BINARY_DIR}
    OV_RUNTIME_LIBS_DIR=${OV_WHEEL_RUNTIMEDIR}
    TBB_LIBS_DIR=${TBB_LIBS_DIR}
    PUGIXML_LIBS_DIR=${PUGIXML_LIBS_DIR}
    PY_PACKAGES_DIR=${PY_PACKAGES_DIR})

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${OpenVINO_SOURCE_DIR}/licensing" "${CMAKE_BINARY_DIR}/licensing"
    COMMAND ${wheel_env}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
            --quiet
            --no-user-cfg
            bdist_wheel
            --dist-dir ${openvino_wheels_output_dir}
            --build-number=${WHEEL_BUILD}
            --plat-name=${PLATFORM_TAG}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${wheel_env}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" clean
    DEPENDS ${openvino_wheel_deps}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

set(fdupes_report ${CMAKE_CURRENT_BINARY_DIR}/fdupes_report.txt)
add_custom_command(OUTPUT "${fdupes_report}"
    COMMAND ${CMAKE_COMMAND}
        -D PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -D WORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        -D WHEEL_VERSION=${WHEEL_VERSION}
        -D PACKAGE_FILE=${openvino_wheel_path}
        -D REPORT_FILE=${fdupes_report}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/fdupes_check.cmake"
    DEPENDS "${openvino_wheel_path}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Run 'fdupes' checks for wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel ALL DEPENDS ${openvino_wheel_path} ${fdupes_report})

# install

ie_cpack_add_component(${OV_CPACK_COMP_PYTHON_WHEELS} HIDDEN)

install(FILES ${openvino_wheel_path}
        DESTINATION ${OV_CPACK_WHEELSDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_WHEELS})
