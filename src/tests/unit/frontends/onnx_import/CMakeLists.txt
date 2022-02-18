# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME onnxImporterUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            gtest
            gtest_main
            commonTestUtils
            openvino_onnx_frontend
            openvino::util
        DEFINES
            ONNX_MODELS_DIR=\"${TEST_MODEL_ZOO}/onnx_import\"
        ADD_CPPLINT
        DEPENDENCIES
            test_model_zoo
        LABELS
            ONNX
)
