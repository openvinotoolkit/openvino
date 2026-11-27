# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_subgraphs_dumper_tests)

include(sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES ${OV_SUBGRAPHS_DUMPER_TEST_SOURCES}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/
        LINK_LIBRARIES
            PRIVATE
            ov_subgraphs_dumper_util
            func_test_utils
        LABELS
            OV UNIT
)

ov_build_target_faster(${TARGET_NAME} UNITY)
