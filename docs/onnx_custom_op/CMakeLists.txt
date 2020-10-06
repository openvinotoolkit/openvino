# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:onnx_custom_op]
set(TARGET_NAME "onnx_custom_op")

find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)

add_library(${TARGET_NAME} STATIC onnx_custom_op.cpp)

target_compile_options(${TARGET_NAME} PRIVATE -std=c++11)

target_include_directories(${TARGET_NAME} PRIVATE ${InferenceEngine_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} PRIVATE IE::inference_engine_legacy IE::inference_engine
                                             ${NGRAPH_LIBRARIES} ${ONNX_IMPORTER_LIBRARIES})
# [cmake:onnx_custom_op]
