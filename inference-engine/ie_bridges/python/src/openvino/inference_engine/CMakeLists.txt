# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "ie_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/inference_engine)

file(GLOB SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${SOURCE} PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCE})

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 11 LINKER_LANGUAGE CXX)
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/inference_engine/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/requirements.txt ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../../requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../__init__.py
)