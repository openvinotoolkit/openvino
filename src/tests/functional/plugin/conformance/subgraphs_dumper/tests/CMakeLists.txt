# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumperTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src
        EXCLUDED_SOURCE_PATHS
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src/main.cpp
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include
            ${CMAKE_CURRENT_SOURCE_DIR}/
        LINK_LIBRARIES
            PRIVATE
                func_test_utils
                openvino::runtime
                openvino::pugixml
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)
