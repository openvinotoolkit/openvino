# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME multiFuncTests)

set(DEPS "")
if (ENABLE_MKL_DNN)
    list(APPEND DEPS MKLDNNPlugin)
endif()
if (ENABLE_CLDNN)
    list(APPEND DEPS clDNNPlugin)
endif()
if (ENABLE_MYRIAD)
    list(APPEND DEPS myriadPlugin)
endif()

addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            MultiDevicePlugin
            ${DEPS}
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            MULTI
)