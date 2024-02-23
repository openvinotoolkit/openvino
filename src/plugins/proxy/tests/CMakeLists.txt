# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_proxy_plugin_tests)

set(DEPENDENCIES
    mock_engine
)

set(COMPILE_DEFINITIONS "CI_BUILD_NUMBER=\"mock_version\"")

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO_BATCH)
endif()

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
    list(APPEND COMPILE_DEFINITIONS HETERO_ENABLED)
endif()

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
    list(APPEND COMPILE_DEFINITIONS IR_FRONTEND_ENABLED)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            openvino::runtime::dev
            gtest
            gtest_main
            common_test_utils
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT PROXY
)

target_compile_definitions(${TARGET_NAME} PRIVATE ${COMPILE_DEFINITIONS})
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../dev_api)
