# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PRIVATE
                gflags
                inference_engine
                commonTestUtils
                pugixml::static
        ADD_CPPLINT
)

if(TARGET ov_ir_frontend)
    add_dependencies(${TARGET_NAME} ov_ir_frontend)
endif()

ie_faster_build(${TARGET_NAME} UNITY)
