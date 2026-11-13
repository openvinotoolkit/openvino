# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_lp_transformations_tests)

include(sources.cmake)

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES
        ${LP_TRANSFORMATIONS_TESTS_SRCS}
    DEPENDENCIES
    LINK_LIBRARIES
        gtest
        gtest_main
        openvino::runtime::dev
        common_test_utils
        ov_lpt_models
        gmock
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS
        OV UNIT LP_TRANSFORMATIONS
    CHECK_SOURCES_LISTED
)

ov_build_target_faster(${TARGET_NAME} PCH)
