# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "ie_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)

file(GLOB SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/ie_api.pyx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${SOURCE} PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCE})
set(INSTALLED_TARGETS ${TARGET_NAME})

file(GLOB OTHER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx)
list(REMOVE_ITEM OTHER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ie_api.pyx")

foreach(PYX_FILE ${OTHER_SOURCES})
    get_filename_component(PYX_NAME "${PYX_FILE}" NAME_WE)
    set_source_files_properties(${PYX_FILE} PROPERTIES CYTHON_IS_CXX ON)
    cython_add_module(${PYX_NAME} ${PYX_FILE})
    add_dependencies(${TARGET_NAME} ${PYX_NAME})
    target_include_directories(${PYX_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${PYX_NAME} PRIVATE ${InferenceEngine_LIBRARIES})
    list(APPEND INSTALLED_TARGETS ${PYX_NAME})
endforeach()

if(COMMAND ie_add_vs_version_file)
    foreach(target IN LISTS INSTALLED_TARGETS)
        ie_add_vs_version_file(NAME ${target}
                               FILEDESCRIPTION "Inference Engine Python library")
    endforeach()
endif()

function(python_disable_deprecated_warnings)
    disable_deprecated_warnings()
    set(pyx_file "${CMAKE_CURRENT_BINARY_DIR}/ie_api.cxx" "${CMAKE_CURRENT_BINARY_DIR}/constants.cxx")
    set_source_files_properties(${pyx_file} PROPERTIES COMPILE_FLAGS ${ie_c_cxx_deprecated})
endfunction()

python_disable_deprecated_warnings()

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/requirements.txt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../../requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../__init__.py
)

# creates a folder in openvino directory and a symlink to benchmark
# inside bin directory for developers for running python benchmark_app
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../tools
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${OpenVINO_MAIN_SOURCE_DIR}/tools/benchmark  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../tools/benchmark
)

# install

install(TARGETS ${INSTALLED_TARGETS}
        RUNTIME DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION}
        ARCHIVE DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION}
        LIBRARY DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}/openvino/inference_engine COMPONENT ${PYTHON_VERSION})

install(PROGRAMS __init__.py
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${PYTHON_VERSION}/openvino/inference_engine
        COMPONENT ${PYTHON_VERSION})
