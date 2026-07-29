# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_util_tests)

set(${TARGET_NAME}_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common_util_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/file_util_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_inputs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_comparator_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ov_tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/specialize_function.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils_tests.cpp
)

ov_add_test_target(
    NAME ${TARGET_NAME}
    SOURCES ${${TARGET_NAME}_SOURCES}
    LINK_LIBRARIES
        common_test_utils
        func_test_utils
        base_func_tests
    LABELS OV UNIT
    GTEST_DISCOVER
    TESTS_PER_SOURCE
)
