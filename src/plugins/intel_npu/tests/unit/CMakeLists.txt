# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_npu_unit_tests")

# FIXME: All those dependencies below need a huge rework!
# All of them are required to test complex (in terms of dependencies) classes,
# such as npuw::CompiledModel or npuw::JustSyncInferRequest.

set(MANDATORY_UNIT_TESTS_LIBS
        "openvino::commonTestUtils"
        "openvino::gmock"
        "openvino::gtest"
        "openvino::gtest_main"
        "openvino::runtime"
        "openvino::reference"
        "openvino::npu_common"
        "openvino::npu_al"
        "openvino::npu_logger_utils"
        "openvino_npu_driver_compiler_adapter"
        "openvino_npu_level_zero_backend"
)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/partitioning
        DEPENDENCIES
            openvino::runtime
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw/model_generator
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/compiler_adapter/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw
        OBJECT_FILES
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/metadata.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/plugin.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/remote_context.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/metrics.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/properties.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/async_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/backends_registry.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/logging.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/base_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/just_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/unfold_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/weights_bank.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/serialization.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/spatial.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util_xarch.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/lazy_tensor.cpp
        LINK_LIBRARIES
            ${MANDATORY_UNIT_TESTS_LIBS}
        LABELS
            NPUW
)

if(ENABLE_AVX2)
    ov_avx2_optimization_flags(avx2_flags)
    target_compile_options(${TARGET_NAME} PRIVATE "${avx2_flags}")
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_AVX2)
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER=\"mock_version\")

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
