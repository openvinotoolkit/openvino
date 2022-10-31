# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_transformations_tests)

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
            funcTestUtils
            inference_engine_obj
            inference_engine_legacy
            offline_transformations
            sharedTestClasses
        INCLUDES
            $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
        ADD_CLANG_FORMAT
        LABELS
            TRANSFORMATIONS
)
