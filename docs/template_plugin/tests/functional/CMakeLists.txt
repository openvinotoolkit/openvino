# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    ie_enable_lto()
endif()

# [cmake:functional_tests]
set(TARGET_NAME templateFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            templatePlugin
        LINK_LIBRARIES
            IE::funcSharedTests
        INCLUDES
            "${IE_MAIN_TEMPLATE_PLUGIN_SOURCE_DIR}/include"
        ADD_CPPLINT
        LABELS
            TEMPLATE
)
# [cmake:functional_tests]
