# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_api_conformance_tests)

include(sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES ${OV_API_CONFORMANCE_TEST_SOURCES}
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                conformance_shared
        LABELS
            OV API_CONFORMANCE
)

ov_build_target_faster(${TARGET_NAME} UNITY)
