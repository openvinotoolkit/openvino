# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_tests)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            openvino_ir_frontend
            mock_engine
            openvino_hetero_plugin
            template_extension
            openvino_template_extension
            openvino_auto_plugin
            openvino_auto_batch_plugin
        LINK_LIBRARIES
            gtest
            gmock
            gtest_main
            openvino::runtime::dev
            commonTestUtils
            inference_engine_legacy
            inference_engine_obj
            funcTestUtils
        INCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
        ADD_CLANG_FORMAT
        LABELS
            OV
)

check_cxx_compiler_flag("-Wno-suggest-override" SUGGEST_OVERRIDE_SUPPORTED)
if(SUGGEST_OVERRIDE_SUPPORTED)
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-suggest-override")
endif()

