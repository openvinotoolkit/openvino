# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformanceShared)

addIeTargetTest(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        DEVELOPER_PACKAGE
            inference_engine_tests
        ADD_CPPLINT
        INCLUDES
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                gflags
                funcSharedTests
        LABELS
            CONFORMANCE
)

ie_faster_build(${TARGET_NAME} UNITY)
