# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME onnx
                LINKABLE_FRONTEND
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert ONNX file format"
                LINK_LIBRARIES ngraph::builder openvino::util onnx_common inference_engine_transformations)

set(ONNX_OPSET_VERSION 13 CACHE INTERNAL "Supported version of ONNX operator set")
target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp")
