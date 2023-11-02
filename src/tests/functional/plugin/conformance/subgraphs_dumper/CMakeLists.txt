# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OLD_TARGET_NAME subgraphsDumper)
set(TARGET_NAME ov_subgraphs_dumper)

list(APPEND LIBRARIES
        gflags
        func_test_utils
        op_conformance_utils
)

# add subgraphs_dumpers tool
ov_add_test_target(
        NAME ${TARGET_NAME}
        TYPE EXECUTABLE
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
        INCLUDES
            PRIVATE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        ADD_CPPLINT
)

# todo: remove the changes as the product config will be changed
# add subgraphs_dumpers tool
ov_add_test_target(
        NAME ${OLD_TARGET_NAME}
        TYPE EXECUTABLE
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
        INCLUDES
            PRIVATE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        ADD_CPPLINT
)


# add subgraphs_dumpers lib to get API
ov_add_target(
        NAME "${TARGET_NAME}_util"
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                ${LIBRARIES}
        EXCLUDED_SOURCE_PATHS
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/src/main.cpp
            ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper/include/gflag_config.hpp
        DEPENDENCIES
            ov_frontends
            ${LIBRARIES}
        ADD_CPPLINT
)

ov_build_target_faster(${TARGET_NAME} UNITY)
ov_build_target_faster(${OLD_TARGET_NAME} UNITY)
