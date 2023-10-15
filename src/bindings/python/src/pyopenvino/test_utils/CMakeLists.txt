# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "test_utils_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/test_utils)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cpp)

# create target

pybind11_add_module(${TARGET_NAME} MODULE NO_EXTRAS ${SOURCES})

add_dependencies(${TARGET_NAME} pyopenvino)

if(COMMAND ov_add_vs_version_file)
    ov_add_vs_version_file(NAME ${TARGET_NAME}
            FILEDESCRIPTION "Test Utils Python library")
endif()

if(OpenVINODeveloperPackage_FOUND)
    list(APPEND link_libraries openvino::commonTestUtils)
else()
    list(APPEND link_libraries common_test_utils)
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

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                           EXCLUDE_PATTERNS ".*\\.cxx")

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${OV_CPACK_PYTHONDIR}/openvino/test_utils
        COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${OV_CPACK_PYTHONDIR}/tests/openvino/test_utils
        COMPONENT tests EXCLUDE_FROM_ALL)

install(PROGRAMS ${OpenVINOPython_SOURCE_DIR}/src/openvino/test_utils/__init__.py
        DESTINATION ${OV_CPACK_PYTHONDIR}/tests/openvino/test_utils
        COMPONENT tests
        EXCLUDE_FROM_ALL)
