# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            MKLDNNPlugin
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            CPU
)

target_compile_definitions(${TARGET_NAME} PRIVATE IR_SERIALIZATION_MODELS_PATH="${CMAKE_SOURCE_DIR}/inference-engine/tests/functional/inference_engine/ir_serialization/models/")
