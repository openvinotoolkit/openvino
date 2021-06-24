# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformanceTests)

list(APPEND EXPORT_DEPENDENCIES
        gflags
        funcSharedTests
    )

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        INCLUDES
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                ${EXPORT_DEPENDENCIES}
        DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
        LABELS
            CONFORMANCE
)

ie_faster_build(${TARGET_NAME} UNITY)
