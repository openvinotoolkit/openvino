# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_lp_transformations_tests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
    LINK_LIBRARIES
        gtest
        gtest_main
        openvino::runtime::dev
        commonTestUtils
        lptNgraphFunctions
        gmock
    ADD_CLANG_FORMAT
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS
        LP_TRANSFORMATIONS
)