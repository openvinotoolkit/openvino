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

# add subgraphs_dumpers tool
ov_add_target(
        NAME ${TARGET_NAME}
        TYPE EXECUTABLE
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

# add subgraphs_dumpers lib to get API
ov_add_target(
        NAME "${TARGET_NAME}Util"
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        EXCLUDED_SOURCE_PATHS
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src/main.cpp
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include/gflag_config.hpp
        DEPENDENCIES
            ov_frontends
        ADD_CPPLINT
)

ov_build_target_faster(${TARGET_NAME} UNITY)