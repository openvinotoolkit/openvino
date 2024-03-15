# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME onnx
                LINKABLE_FRONTEND
                PROTOBUF_REQUIRED
                PROTOBUF_LITE
                SKIP_NCC_STYLE
                FILEDESCRIPTION "FrontEnd to load and convert ONNX file format"
                LINK_LIBRARIES openvino_onnx_common openvino::core::dev)

set(ONNX_OPSET_VERSION 18 CACHE INTERNAL "Supported version of ONNX operator set")
target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES "${${TARGET_NAME}_INCLUDE_DIR}"
                    DEFINITIONS
                        $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)
