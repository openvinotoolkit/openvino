# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_subgraphs_dumper_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/
        LINK_LIBRARIES
            PRIVATE
            ov_subgraphs_dumper_util
            func_test_utils
        ADD_CPPLINT
        LABELS
            OV UNIT
)

ov_build_target_faster(${TARGET_NAME} UNITY)
