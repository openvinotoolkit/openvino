# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieIRV7Tests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            funcTestUtils
        ADD_CPPLINT
        DEPENDENCIES
            inference_engine_ir_reader
        LABELS
            IE
)
