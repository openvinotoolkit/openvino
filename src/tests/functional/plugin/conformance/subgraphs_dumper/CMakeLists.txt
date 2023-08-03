# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

list(APPEND LIBRARIES
        gflags
        openvino::runtime
        func_test_utils
        openvino::pugixml
)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
        INCLUDES
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ov_frontends
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)