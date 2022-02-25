# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

list(APPEND DEPENDENCIES
        gflags
        inference_engine
        commonTestUtils
        pugixml::static
)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PRIVATE
                ${DEPENDENCIES}
        DEPENDENCIES
            ${DEPENDENCIES}
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
