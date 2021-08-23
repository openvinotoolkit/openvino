# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "template_operations")

find_package(ngraph REQUIRED)
find_package(OpenCV QUIET COMPONENTS core)

set(SRC op.hpp op.cpp)

if (OpenCV_FOUND)
    set(SRC ${SRC} fft_op.hpp fft_op.cpp)
endif()

add_library(${TARGET_NAME} STATIC ${SRC})

if (OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENCV_IMPORT_ENABLED)
    target_link_libraries(${TARGET_NAME} PUBLIC opencv_core)
endif()

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ngraph::ngraph)
# [cmake:extension]

# Enable code style check
file(GLOB_RECURSE template_operations_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${template_operations_src})
