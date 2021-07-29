# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:onnx_custom_op]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "onnx_custom_op")

find_package(ngraph REQUIRED COMPONENTS onnx_importer)

add_library(${TARGET_NAME} STATIC onnx_custom_op.cpp onnx_custom_op.hpp)

target_link_libraries(${TARGET_NAME} PUBLIC ${NGRAPH_LIBRARIES} ${ONNX_IMPORTER_LIBRARIES})
# [cmake:onnx_custom_op]

# Enable code style check
add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
