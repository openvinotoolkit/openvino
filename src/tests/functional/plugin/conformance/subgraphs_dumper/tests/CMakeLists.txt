# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumperTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            # TODO: create static library for matchers instead of duplication
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src/matchers
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include
        LINK_LIBRARIES
            PRIVATE
                funcTestUtils
                openvino::runtime
                openvino::pugixml
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
