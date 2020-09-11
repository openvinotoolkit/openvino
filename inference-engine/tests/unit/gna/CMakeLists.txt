# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
            GNAPlugin_test_static
        ADD_CPPLINT
        LABELS
            GNA
)