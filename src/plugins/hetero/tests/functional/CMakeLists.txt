# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_func_tests)
ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            mock_engine
            openvino_hetero_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
            openvino::runtime::dev
            gtest
            gtest_main
            common_test_utils
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_hetero_plugin,SOURCE_DIR>/src
        ADD_CLANG_FORMAT
        LABELS
            OV HETERO
)

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(${TARGET_NAME} openvino_ir_frontend)
    target_compile_definitions(${TARGET_NAME} PRIVATE IR_FRONTEND_ENABLED)
endif()
