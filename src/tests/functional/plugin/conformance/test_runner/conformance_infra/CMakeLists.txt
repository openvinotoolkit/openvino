# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conformanceShared)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
                ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper_new/include/cache/meta/
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        INCLUDES
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/conformance/subgraphs_dumper_new/include/cache/meta/
        LINK_LIBRARIES
            PUBLIC
                gflags
                funcSharedTests
)

ie_faster_build(${TARGET_NAME} UNITY)
