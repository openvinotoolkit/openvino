# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT NGRAPH_USE_PROTOBUF_LITE)
    set(TARGET_NAME onnxImporterUnitTests)

    addIeTargetTest(
            NAME ${TARGET_NAME}
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDENCIES
                ngraph
                onnx_importer
            LINK_LIBRARIES
                unitTestUtils
                onnx_importer
            DEFINES
                ONNX_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/models\"
            ADD_CPPLINT
            LABELS
                ONNX
    )
endif()
