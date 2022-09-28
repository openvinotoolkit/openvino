# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME onnx
                LINKABLE_FRONTEND
                PROTOBUF_LITE
                SKIP_NCC_STYLE
                FILEDESCRIPTION "FrontEnd to load and convert ONNX file format"
                LINK_LIBRARIES ngraph::builder openvino::util onnx_common openvino::runtime::dev)

set(ONNX_OPSET_VERSION 16 CACHE INTERNAL "Supported version of ONNX operator set")
target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORY "${${TARGET_NAME}_INCLUDE_DIR}"
                    DEFINITIONS
                        $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:frontend_common::static,INTERFACE_INCLUDE_DIRECTORIES>)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_import
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}/ngraph/frontend
        COMPONENT ${OV_CPACK_COMP_CORE_DEV})
