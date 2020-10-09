# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:extension]
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "template_extension")

find_package(ngraph REQUIRED OPTIONAL_COMPONENTS onnx_importer)
find_package(InferenceEngine REQUIRED)

file(GLOB_RECURSE SRC *.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
set(LIBRARIES IE::inference_engine_legacy IE::inference_engine ${NGRAPH_LIBRARIES})
if (ngraph_onnx_importer_FOUND)
    list(APPEND LIBRARIES ${ONNX_IMPORTER_LIBRARIES})
    target_compile_definitions(${TARGET_NAME} PRIVATE NGRAPH_ONNX_IMPORT_ENABLED)
endif()
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBRARIES})
# [cmake:extension]
