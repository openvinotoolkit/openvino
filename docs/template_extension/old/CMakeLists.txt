# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "template_extension")

find_package(OpenVINO REQUIRED COMPONENTS Runtime OPTIONAL_COMPONENTS ONNX)
find_package(OpenCV QUIET COMPONENTS core)

set(SRC cpu_kernel.cpp extension.cpp op.cpp)

if(OpenCV_FOUND)
    set(SRC ${SRC} fft_kernel.cpp fft_op.cpp)
endif()

add_library(${TARGET_NAME} MODULE ${SRC})

if(OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_IMPORT_ENABLED)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

if(OpenVINO_Frontend_ONNX_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE openvino::frontend::onnx)
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_ONNX_FRONTEND_ENABLED)
endif()
# [cmake:extension]

# Enable code style check
file(GLOB_RECURSE template_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${template_extension_src})
