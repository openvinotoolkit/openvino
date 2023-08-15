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
                "${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include/cache/meta/"
        ADD_CPPLINT
        LINK_LIBRARIES
            PUBLIC
                conformanceShared
        LABELS
            OP_CONFORMANCE
)

ie_faster_build(${TARGET_NAME} UNITY)
