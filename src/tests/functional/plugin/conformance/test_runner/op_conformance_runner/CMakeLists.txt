# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_op_conformance_tests)

include(sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES ${OV_OP_CONFORMANCE_TEST_SOURCES}
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        LINK_LIBRARIES
            PUBLIC
                conformance_shared
                op_conformance_utils
        LABELS
            OV OP_CONFORMANCE
)

ov_build_target_faster(${TARGET_NAME} UNITY)
