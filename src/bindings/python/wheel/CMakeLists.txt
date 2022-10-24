# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common vars used by setup.py
#

set(pyversion python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
set(WHEEL_VERSION "${OpenVINO_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)

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

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')"
                OUTPUT_VARIABLE PYTHON_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')"
                OUTPUT_VARIABLE ABI_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags._platform_tags())}')"
                OUTPUT_VARIABLE PLATFORM_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)

macro(_ov_platform_arch)
    if(AARCH64)
        set(_arch "aarch64")
    elseif(ARM)
        set(_arch "armvl7")
    elseif(X86_64)
        set(_arch "x86_64")
    elseif(X86)
        set(_arch "i686")
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

    execute_process(COMMAND ldd --version
        OUTPUT_VARIABLE _libc_version
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_libc_version MATCHES ".*([0-9]+)\\.([0-9]+).*")
        set(_libc_major "${CMAKE_MATCH_1}")
        set(_libc_minor "${CMAKE_MATCH_2}")
    endif()

    # common pattern manylinux_<libc major>_<libc minor>_<arch>
    if(_libc_version AND _arch)
        set(PLATFORM_TAG "manylinux_${_libc_major}_${_libc_minor}_${_arch}")
    endif()
endif()

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

#
# create target for openvino.wheel
#

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${OpenVINO_SOURCE_DIR}/licensing" "${CMAKE_BINARY_DIR}/licensing"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${CMAKE_COMMAND} -E env
            WHEEL_VERSION=${WHEEL_VERSION}
            WHEEL_BUILD=${WHEEL_BUILD}
            OPENVINO_BUILD_DIR=${OpenVINO_BINARY_DIR}
            OPENVINO_PYTHON_BUILD_DIR=${OpenVINOPython_BINARY_DIR}
            OV_RUNTIME_LIBS_DIR=${OV_WHEEL_RUNTIMEDIR}
            TBB_LIBS_DIR=${TBB_LIBS_DIR}
            PUGIXML_LIBS_DIR=${PUGIXML_LIBS_DIR}
            PY_PACKAGES_DIR=${PY_PACKAGES_DIR}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" clean bdist_wheel
            --dist-dir ${openvino_wheels_output_dir}
            --build-number=${WHEEL_BUILD}
            --plat-name=${PLATFORM_TAG}
            --quiet
    DEPENDS ${openvino_wheel_deps}
           "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel ALL DEPENDS ${openvino_wheel_path})

# install

ie_cpack_add_component(${OV_CPACK_COMP_PYTHON_WHEELS} HIDDEN)

install(FILES ${openvino_wheel_path}
        DESTINATION ${OV_CPACK_WHEELSDIR}
        COMPONENT ${OV_CPACK_COMP_PYTHON_WHEELS})
