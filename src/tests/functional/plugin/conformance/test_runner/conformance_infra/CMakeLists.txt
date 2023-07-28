# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformanceShared)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        INCLUDES
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                gflags
                funcSharedTests
)

if(ENABLE_CONFORMANCE_PGQL)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_CONFORMANCE_PGQL)
endif()

ie_faster_build(${TARGET_NAME} UNITY)
