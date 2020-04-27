# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

set(TARGET_NAME vpuUnitTests)

include(${XLINK_DIR}/XLink.cmake)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/vpu/myriad_plugin
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include/watchdog
            ${XLINK_INCLUDE}
            ${XLINK_PLATFORM_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/base/
        LINK_LIBRARIES
            vpu_graph_transformer_test_static
            unitTestUtils
            mvnc
        ADD_CPPLINT
        LABELS
            VPU
            MYRIAD
)