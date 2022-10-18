# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "openvino_template_extension")

find_package(OpenVINO REQUIRED)

set(SRC identity.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
# [cmake:extension]
