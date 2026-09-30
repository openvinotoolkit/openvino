# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_tensorflow_common_tests")

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES
    LINK_LIBRARIES
    gtest_main_manifest
    frontend_shared_test_classes
    openvino_tensorflow_common
    ADD_CLANG_FORMAT
    CHECK_SOURCES_LISTED
    LABELS
    OV UNIT TF_COMMON
)
