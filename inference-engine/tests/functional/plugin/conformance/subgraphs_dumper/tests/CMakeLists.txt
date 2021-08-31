# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumperTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS ${OpenVINO_SOURCE_DIR}/inference-engine/tests/functional/plugin/conformance/subgraphs_dumper/src/matchers
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/inference-engine/tests/functional/plugin/conformance/subgraphs_dumper/include
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
        LINK_LIBRARIES
            PRIVATE
                funcTestUtils
                ngraph
                pugixml::static
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
