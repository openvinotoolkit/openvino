# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_tensorflow_common_tests")

ov_add_test_target(
    NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            gtest_main_manifest 
            frontend_shared_test_classes
            openvino_tensorflow_common
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT TF_COMMON
)
