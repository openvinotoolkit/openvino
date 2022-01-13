# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "py_paddle_frontend")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/paddle)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/paddle)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/paddle)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/frontend/paddle)
set(PYTHON_BRIDGE_CPACK_PATH "python")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# create target

pybind11_add_module(${TARGET_NAME} MODULE ${SOURCES})

add_dependencies(${TARGET_NAME} pyopenvino)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime ov_paddle_frontend)

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy  ${PYTHON_SOURCE_DIR}/openvino/frontend/paddle/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
        )

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
        EXCLUDE_PATTERNS ".*\\.cxx")

# install

#ie_python_minimal_api(${TARGET})
#add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})

#ie_cpack_add_component(pyopenvino_${PYTHON_VERSION})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../openvino/frontend/paddle
        DESTINATION python/${PYTHON_VERSION}
        COMPONENT pyopenvino_${PYTHON_VERSION}
        USE_SOURCE_PERMISSIONS)

install(TARGETS ${TARGET}
        DESTINATION python/${PYTHON_VERSION}/openvino/frontend/paddle
        COMPONENT pyopenvino_${PYTHON_VERSION})
