# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME autoRealResCheck)

addIeTargetTest(
        NAME
        ${TARGET_NAME}
        ROOT
        ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        openvino_auto_plugin
        LINK_LIBRARIES
        funcSharedTests
        ADD_CPPLINT
        LABELS
            GPU
            CPU
)

set(TEST_AUTO_MODELS_DIRNAME /home/bell)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_AUTO_MODELS_DIRNAME=\"${TEST_AUTO_MODELS_DIRNAME}/\")
