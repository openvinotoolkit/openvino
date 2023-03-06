# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_transformations_tests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
    LINK_LIBRARIES
        gmock
        funcTestUtils
        offline_transformations
        sharedTestClasses
        lptNgraphFunctions
    ADD_CLANG_FORMAT
    INCLUDES
        $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
    LABELS
        TRANSFORMATIONS
)

if(WIN32)
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()
