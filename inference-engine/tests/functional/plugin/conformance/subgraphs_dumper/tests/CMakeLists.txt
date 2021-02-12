# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumperTests)

list(APPEND DEPENDENCIES
        unitTestUtils
        ngraph
        )

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../src/matchers
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${CMAKE_ROOT}/inference_engine/src/transformations/include
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            PRIVATE
                ${DEPENDENCIES}
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME}
        PCH PRIVATE "../src/precomp.hpp"
        )
#set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
