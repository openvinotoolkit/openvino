# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME sharedTestClasses)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        INCLUDES
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PRIVATE
                pugixml::static
                funcTestUtils
                ngraphFunctions
                inference_engine_legacy
)

ie_faster_build(${TARGET_NAME}
        PCH PRIVATE "src/precomp.hpp"
        )
