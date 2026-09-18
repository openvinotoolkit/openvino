# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_transformations_tests)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-missing-declarations)
endif()

include(sources.cmake)

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES
        ${OV_TRANSFORMATIONS_TESTS_SRCS}
    DEPENDENCIES
    LINK_LIBRARIES
        gmock
        func_test_utils
        openvino::offline_transformations
        openvino::reference
        ov_lpt_models
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT TRANSFORMATIONS
    CHECK_SOURCES_LISTED
    CHECK_SOURCES_EXCLUDE_FILES
        ${OV_TRANSFORMATIONS_TESTS_CHECK_SOURCES_EXCLUDE_FILES}
)

ov_build_target_faster(${TARGET_NAME} PCH)
