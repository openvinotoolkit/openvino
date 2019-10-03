# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "statistics_collector_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/tools/statistics_collector)

file(GLOB SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.pyx)

set_source_files_properties(${SOURCE} PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCE})

set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 11 LINKER_LANGUAGE CXX)

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-register")
endif()

if(TARGET IE::statistics_collector_s)
	target_link_libraries(${TARGET_NAME} PRIVATE IE::statistics_collector_s)
else()
	target_link_libraries(${TARGET_NAME} PRIVATE statistics_collector_s)
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/tools/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/tools/statistics_collector/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
 )
