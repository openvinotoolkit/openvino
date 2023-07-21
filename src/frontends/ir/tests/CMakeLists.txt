# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_ir_frontend_tests)

list(APPEND IR_TESTS_DEPENDENCIES openvino_ir_frontend openvino_intel_cpu_plugin)
if (NOT UNIX OR NOT ENABLE_INTEL_CPU)
    list(REMOVE_ITEM IR_TESTS_DEPENDENCIES openvino_intel_cpu_plugin)
    set(EXCLUDED_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/frontend_test_mmap.cpp)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_TESTS}
        DEPENDENCIES
            ${IR_TESTS_DEPENDENCIES}
        LINK_LIBRARIES
            gtest
            gtest_main
            openvino::runtime::dev
            common_test_utils
        INCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        ADD_CLANG_FORMAT
        LABELS
            OV
            IR_FE
)
