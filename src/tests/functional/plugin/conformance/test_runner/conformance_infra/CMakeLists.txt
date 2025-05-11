# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformance_shared)

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
                ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper_new/include/cache/meta/
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        LINK_LIBRARIES
            PUBLIC
                gflags
                funcSharedTests
)

if(ENABLE_CONFORMANCE_PGQL)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_CONFORMANCE_PGQL)
endif()

ov_build_target_faster(${TARGET_NAME} UNITY)
