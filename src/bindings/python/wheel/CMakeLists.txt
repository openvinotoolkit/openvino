# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common vars used by setup.py
#

set(pyversion python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
set(WHEEL_VERSION "${OpenVINO_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)

set(PYTHON_BRIDGE_CPACK_PATH "${OV_CPACK_PYTHONDIR}")
set(PY_PACKAGES_DIR ${PYTHON_BRIDGE_CPACK_PATH}/${pyversion})
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

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')"
                OUTPUT_VARIABLE PYTHON_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')"
                OUTPUT_VARIABLE INTERPRETER OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags._platform_tags())}')"
                OUTPUT_VARIABLE WHEEL_PLATFORM OUTPUT_STRIP_TRAILING_WHITESPACE)

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${INTERPRETER}-${PYTHON_ABI}-${WHEEL_PLATFORM}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

#
# create target for openvino.wheel
#

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${OpenVINO_SOURCE_DIR}/licensing" "${CMAKE_BINARY_DIR}/licensing"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${CMAKE_COMMAND} -E env WHEEL_VERSION=${WHEEL_VERSION}
            WHEEL_BUILD=${WHEEL_BUILD}
            OPENVINO_BUILD_DIR=${OpenVINO_BINARY_DIR}
            OPENVINO_PYTHON_BUILD_DIR=${OpenVINOPython_BINARY_DIR}
            OV_RUNTIME_LIBS_DIR=${OV_WHEEL_RUNTIMEDIR}
            TBB_LIBS_DIR=${TBB_LIBS_DIR}
            PUGIXML_LIBS_DIR=${PUGIXML_LIBS_DIR}
            PY_PACKAGES_DIR=${PY_PACKAGES_DIR}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" clean bdist_wheel
            --dist-dir ${openvino_wheels_output_dir}
            --build=${WHEEL_BUILD}
            --plat-name=${WHEEL_PLATFORM}
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
