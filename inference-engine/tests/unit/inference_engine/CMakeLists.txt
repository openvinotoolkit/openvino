# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
            inference_engine_ir_reader_v7
        ADD_CPPLINT
        DEPENDENCIES
            mock_engine
        LABELS
            IE
)
