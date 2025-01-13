# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 17)

set(TARGET_NAME "openvino_template_extension")

# The OpenVINO installed from PyPI can be used to find OpenVINO_DIR
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
            OUTPUT_VARIABLE OpenVINO_DIR_PY
            ERROR_QUIET)
    endif()
endif()

find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

set(SRC identity.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# [cmake:extension]
install(TARGETS ${TARGET_NAME}
    LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

# Enable code style check
file(GLOB_RECURSE template_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
ov_add_clang_format_target(openvino_template_extension_clang FOR_SOURCES ${template_extension_src})
