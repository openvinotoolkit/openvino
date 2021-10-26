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