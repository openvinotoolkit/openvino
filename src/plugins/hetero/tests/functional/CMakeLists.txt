# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_func_tests)

include(sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_HETERO_FUNC_TESTS_SRC}
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
        CHECK_SOURCES_LISTED
        LABELS
            OV HETERO
)

target_sources(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:openvino_hetero_plugin,SOURCE_DIR>/src/blob_serialization.cpp
)

if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()
if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

if(ENABLE_OV_IR_FRONTEND)
    add_dependencies(${TARGET_NAME} openvino_ir_frontend)
    target_compile_definitions(${TARGET_NAME} PRIVATE IR_FRONTEND_ENABLED)
endif()
