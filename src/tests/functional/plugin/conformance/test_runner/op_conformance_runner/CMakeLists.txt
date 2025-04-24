# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_op_conformance_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
        INCLUDES
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        LINK_LIBRARIES
            PUBLIC
                conformance_shared
                op_conformance_utils
        LABELS
            OV OP_CONFORMANCE
)

ov_build_target_faster(${TARGET_NAME} UNITY)
