# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_utils_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_api.pyx
                  ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_api_impl.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_api_impl.hpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_api_impl_defs.pxd)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_utils_api.pyx
                            PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCES})
add_dependencies(${TARGET_NAME} ie_api)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "Test Utils Python library")
endif()

if(TARGET commonTestUtils)
    list(APPEND InferenceEngine_LIBRARIES commonTestUtils)
else()
    list(APPEND InferenceEngine_LIBRARIES IE::commonTestUtils)
endif()

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine")
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/test_utils/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ".*\\.cxx;.*\\.pxd;.*\\.pyx")