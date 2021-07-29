# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT NGRAPH_USE_PROTOBUF_LITE)
    set(TARGET_NAME onnxImporterUnitTests)

    addIeTargetTest(
            NAME ${TARGET_NAME}
            ROOT ${CMAKE_CURRENT_SOURCE_DIR}
            LINK_LIBRARIES
                gtest
                gtest_main
                onnx_importer
            DEFINES
                ONNX_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/models\"
            ADD_CPPLINT
            LABELS
                ONNX
    )
endif()
