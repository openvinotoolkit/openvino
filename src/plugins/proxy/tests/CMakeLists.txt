# Copyright (C) 2018-2026 Intel Corporation
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

set(OV_PROXY_PLUGIN_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_compliance_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/import_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/load_proxy_plugin_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/properties_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy_tests.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/query_model_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_context_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_tensor_tests.cpp
)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_PROXY_PLUGIN_TESTS_SRC}
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            openvino::runtime::dev
            gtest
            gtest_main
            common_test_utils
        ADD_CLANG_FORMAT
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT PROXY
)

target_compile_definitions(${TARGET_NAME} PRIVATE ${COMPILE_DEFINITIONS})
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../dev_api)
