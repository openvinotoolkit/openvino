# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::core::dev openvino::frontend::tensorflow_common)

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORY "${${TARGET_NAME}_INCLUDE_DIR}"
                    DEFINITIONS
                        $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:frontend_common::static,INTERFACE_INCLUDE_DIRECTORIES>)

