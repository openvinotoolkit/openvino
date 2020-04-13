# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME onnxImporterUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ngraph
        LINK_LIBRARIES
            unitTestUtils
        DEFINES
            ONNX_MODELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/models\"
        ADD_CPPLINT
        LABELS
            ONNX
)