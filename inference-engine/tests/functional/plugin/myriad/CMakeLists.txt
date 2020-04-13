# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME myriadFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            myriadPlugin
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            VPU
            MYRIAD
)