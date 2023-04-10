# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformanceTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        LINK_LIBRARIES
            PUBLIC
                conformanceShared
        LABELS
            OP_CONFORMANCE
)

if(ENABLE_CONFORMANCE_PGQL)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_CONFORMANCE_PGQL)
endif()

ie_faster_build(${TARGET_NAME} UNITY)
