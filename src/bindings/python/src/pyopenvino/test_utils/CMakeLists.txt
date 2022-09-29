# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_utils_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(PYTHON_BRIDGE_CPACK_PATH "${OV_CPACK_PYTHONDIR}")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp)

# create target

pybind11_add_module(${TARGET_NAME} MODULE NO_EXTRAS ${SOURCES})

add_dependencies(${TARGET_NAME} pyopenvino)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
            FILEDESCRIPTION "Test Utils Python library")
endif()

if(InferenceEngineDeveloperPackage_FOUND)
    list(APPEND link_libraries IE::commonTestUtils)
else()
    list(APPEND link_libraries commonTestUtils)
endif()

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE ${link_libraries})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${OpenVINOPython_SOURCE_DIR}/src/openvino/test_utils/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
        )

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
        EXCLUDE_PATTERNS ".*\\.cxx")

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${pyversion}/openvino/test_utils
        COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${pyversion}/openvino/test_utils
        COMPONENT tests EXCLUDE_FROM_ALL)

install(PROGRAMS ${OpenVINOPython_SOURCE_DIR}/src/openvino/test_utils/__init__.py
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${pyversion}/openvino/test_utils
        COMPONENT tests
        EXCLUDE_FROM_ALL)
