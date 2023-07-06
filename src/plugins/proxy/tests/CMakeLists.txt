# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_proxy_plugin_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino::runtime
            mock_engine
            funcTestUtils
        LINK_LIBRARIES
            openvino::runtime::dev
            gtest
            gtest_main
            funcTestUtils
        ADD_CLANG_FORMAT
        LABELS
        PROXY_PLUGIN
)

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../dev_api)

if(ENABLE_HETERO)
    add_dependencies(${TARGET_NAME} openvino_hetero_plugin)
    target_compile_definitions(${TARGET_NAME} PRIVATE HETERO_ENABLED)
endif()

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(${TARGET_NAME} openvino_ir_frontend)
    target_compile_definitions(${TARGET_NAME} PRIVATE IR_FRONTEND_ENABLED)
endif()
