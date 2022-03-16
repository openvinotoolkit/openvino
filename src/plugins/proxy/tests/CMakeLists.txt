# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_proxy_plugin_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino::runtime
        LINK_LIBRARIES
            openvino::runtime::dev
            gtest
            gtest_main
        ADD_CLANG_FORMAT
        LABELS
        PROXY_PLUGIN
)

if(ENABLE_HETERO)
    add_dependencies(${TARGET_NAME} openvino_hetero_plugin)
endif()
