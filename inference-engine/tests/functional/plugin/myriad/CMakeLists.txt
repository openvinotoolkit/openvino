# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME myriadFuncTests)

disable_deprecated_warnings()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            myriadPlugin
        LINK_LIBRARIES
            vpu_common_lib
            vpu_graph_transformer
            funcSharedTests
        ADD_CPPLINT
        LABELS
            VPU
            MYRIAD
)
