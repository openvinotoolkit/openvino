# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME apiConformanceTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ADD_CPPLINT
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                conformanceShared
        LABELS
            API_CONFORMANCE
)

ie_faster_build(${TARGET_NAME} UNITY)
