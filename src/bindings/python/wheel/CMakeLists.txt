# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(PythonInterp 3 REQUIRED)
set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
set(WHEEL_VERSION "${OpenVINO_VERSION}" CACHE STRING "Version of this release" FORCE)
set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)
set(PYTHON_BRIDGE_CPACK_PATH "python")
set(PY_PACKAGES_DIR ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION})
set(TBB_LIBS_DIR runtime/3rdparty/tbb/lib)
if(WIN32)
    set(TBB_LIBS_DIR runtime/3rdparty/tbb/bin)
endif()

if(LINUX)
    find_host_program(patchelf_program
                      NAMES patchelf
                      DOC "Path to patchelf tool")
    if(NOT patchelf_program)
        message(FATAL_ERROR "patchelf is not found, which is needed to build ie_wheel")
    endif()
endif()

# create target for openvino.wheel

set(openvino_wheel_deps ie_api)
foreach(_target ov_runtime_libraries ie_plugins _pyngraph pyopenvino py_paddle_frontend py_tensorflow_frontend py_onnx_frontend)
    if(TARGET ${_target})
        list(APPEND openvino_wheel_deps ${_target})
    endif()
endforeach()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.bdist_wheel ; print(f'{wheel.bdist_wheel.get_abi_tag()}')" OUTPUT_VARIABLE PYTHON_ABI)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{tags.interpreter_name()}{tags.interpreter_version()}')" OUTPUT_VARIABLE INTERPRETER)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import wheel.vendored.packaging.tags as tags ; print(f'{next(tags._platform_tags())}')" OUTPUT_VARIABLE WHEEL_PLATFORM)
string(STRIP ${PYTHON_ABI} PYTHON_ABI)
string(STRIP ${INTERPRETER} INTERPRETER)
string(STRIP ${WHEEL_PLATFORM} WHEEL_PLATFORM)

set(openvino_wheel_name "openvino-${WHEEL_VERSION}-${WHEEL_BUILD}-${INTERPRETER}-${PYTHON_ABI}-${WHEEL_PLATFORM}.whl")
set(openvino_wheels_output_dir "${CMAKE_BINARY_DIR}/wheels")
set(openvino_wheel_path "${openvino_wheels_output_dir}/${openvino_wheel_name}")

add_custom_command(OUTPUT ${openvino_wheel_path}
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/licensing" "${CMAKE_BINARY_DIR}/licensing"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/site-packages"
    COMMAND ${CMAKE_COMMAND} -E env WHEEL_VERSION=${WHEEL_VERSION}
        WHEEL_BUILD=${WHEEL_BUILD}
        CMAKE_BUILD_DIR=${CMAKE_BINARY_DIR}
        OV_RUNTIME_LIBS_DIR=${IE_CPACK_RUNTIME_PATH}
        TBB_LIBS_DIR=${TBB_LIBS_DIR}
        PY_PACKAGES_DIR=${PY_PACKAGES_DIR}
        ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/setup.py" clean bdist_wheel
        --dist-dir ${openvino_wheels_output_dir}
        --build=${WHEEL_BUILD}
        --plat-name=${WHEEL_PLATFORM}
    DEPENDS ${openvino_wheel_deps} 
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Python wheel ${openvino_wheel_name}"
    VERBATIM)

add_custom_target(ie_wheel ALL DEPENDS ${openvino_wheel_path})

# install

ie_cpack_add_component(python_wheels)

install(FILES ${openvino_wheel_path}
        DESTINATION tools
        COMPONENT python_wheels)
