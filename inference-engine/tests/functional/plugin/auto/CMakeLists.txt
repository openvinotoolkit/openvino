# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME autoFuncTests)

addIeTargetTest(
        NAME
        ${TARGET_NAME}
        ROOT
        ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        MultiDevicePlugin
        LINK_LIBRARIES
        funcSharedTests
        ADD_CPPLINT
        LABELS
            GPU
            CPU
)

set(TEST_AUTO_MODELS_DIRNAME test_model_zoo/auto_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_AUTO_MODELS_DIRNAME=\"${TEST_AUTO_MODELS_DIRNAME}/\")