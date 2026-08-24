# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_conditional_compilation_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/selective_build.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/selective_build_analyzer.cpp
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            openvino::conditional_compilation
        INCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ADD_CLANG_FORMAT
        TESTS_PER_SOURCE
        LABELS
            OV UNIT
)
