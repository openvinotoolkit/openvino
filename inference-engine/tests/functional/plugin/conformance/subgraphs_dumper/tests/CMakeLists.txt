# Copyright (C) 2018-2021 Intel Corporation
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
        ADDITIONAL_SOURCE_DIRS ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/tests/functional/plugin/conformance/subgraphs_dumper/src/matchers
        INCLUDES
            ${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/tests/functional/plugin/conformance/subgraphs_dumper/include
            $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            PRIVATE
                ${DEPENDENCIES}
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
