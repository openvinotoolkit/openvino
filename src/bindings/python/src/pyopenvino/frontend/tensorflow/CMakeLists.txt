# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "py_tensorflow_frontend")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/tensorflow)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/tensorflow)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/tensorflow)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/tensorflow)
set(PYTHON_BRIDGE_CPACK_PATH "python")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# create target

pybind11_add_module(${TARGET_NAME} MODULE ${SOURCES})

add_dependencies(${TARGET_NAME} pyopenvino)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ov_tensorflow_frontend)

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy  ${PYTHON_SOURCE_DIR}/openvino/frontend/tensorflow/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
        )

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
        EXCLUDE_PATTERNS ".*\\.cxx")

install(TARGETS ${TARGET_NAME}
        DESTINATION python/${PYTHON_VERSION}/openvino
        COMPONENT pyopenvino_${PYTHON_VERSION})
