# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_func_tests)

set(OV_HETERO_FUNC_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_model_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hetero_synthetic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hetero_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hetero_tests.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/import_model_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/properties_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/query_model_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_compiled_model/compiled_model_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_compiled_model/import_export.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_compiled_model/properties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/gpu_dyn_batch_shape_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/inference_chaining.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/infer_request_dynamic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/io_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/iteration_chaining.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_infer_request/memory_states.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/caching_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/core_threading_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/life_time.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/properties_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/ov_plugin/version.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/skip_tests_config.cpp
)

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
