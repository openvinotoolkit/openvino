# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/gna_plugin/transformations
        DEPENDENCIES
            GNAPlugin
        LINK_LIBRARIES
            funcSharedTests
            GNAPlugin_test_static
        ADD_CPPLINT
        LABELS
            GNA
)
