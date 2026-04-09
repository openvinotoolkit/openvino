# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_npu_unit_tests")

set(MANDATORY_UNIT_TESTS_LIBS
        "openvino::commonTestUtils"
        "openvino::gmock"
        "openvino::gtest"
        "openvino::gtest_main"
        "openvino::runtime"
        "openvino::reference"
        "openvino::npu_al"
        "openvino::npu_logger_utils"
        "openvino_npu_common"
    "openvino_npu_compiler_adapter"  ## fix /usr/bin/ld: zero_dynamic_pipeline.cpp:(.text._ZN9intel_npu15DynamicPipeline4pushEv+0x26f): undefined reference to `intel_npu::vm_execute_graph(_npu_vm_runtime_handle_t*, std::shared_ptr<intel_npu::ZeroInitStructsHolder> const&, intel_npu::IDynamicGraph::GraphArguments&, std::vector<_ze_command_list_handle_t*, std::allocator<_ze_command_list_handle_t*> >&, _ze_command_queue_handle_t*, _ze_fence_handle_t*, _ze_event_handle_t*)'
        "openvino::npu_ops"
)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/partitioning
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/backend/src
        DEPENDENCIES
            openvino::runtime
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/tests/functional/behavior/npuw/test_engine/models
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/backend/include/
            ${OpenVINO_SOURCE_DIR}/thirdparty/level_zero/level-zero
            ${OpenVINO_SOURCE_DIR}/thirdparty/level_zero/level-zero/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/compiler_adapter/include ## To fix: FAILED: src/plugins/intel_npu/tests/unit/CMakeFiles/ov_npu_unit_tests.dir/__/__/src/backend/src/zero_dynamic_pipeline.cpp.o ... npu_vm_runtime_api.hpp:10:10: fatal error: intel_npu/runtime/npu_vm_runtime.hpp: No such file or directory
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/vm/runtime/include
        OBJECT_FILES
            # plugin/src
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/src/metadata.cpp
            # npuw core
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_infer_base_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_eagle3_extension.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/whisper/whisper_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/whisper/prepare_whisper_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/embedding/embedding_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/embedding/prepare_embedding_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/embedding/redirect_new_kv_to_output.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/embedding/remove_empty_kv_inputs.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/base_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/just_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/unfold_sync_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/logging.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/llm_prefix_caching.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/infer_request_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/serialization.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/util_xarch.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/v1/elements/failsafe.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/lazy_tensor.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pyramid_attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/host_flash_attention.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/perf.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/spatial.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/weights_bank.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/accuracy/comparator.cpp
            # moe
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe/moe_executor.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe/moe_resources.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe/moe_infer_utils.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/moe_transformation.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/moe_unroll_patterns.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/device_routed_moe_transform.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/gather_to_2d_gather.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/moe_transformations/apply_moe_device_routed_transforms.cpp
            # npuw_transformations
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/convert_kvcache_to_precision.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/decompose_gqa.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/lora_stateful_to_stateless.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/optimize_value_tensors.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/patch_phi3_sliding_mask.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/phi3_sliding_mask.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/reshape_sliced_head_to_static.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/reshape_to_static.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/npuw_transformations/slice_out_embeds.cpp
            # kokoro pipeline
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines/kokoro/kokoro_compiled_model.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines/kokoro/kokoro_infer_request.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines/kokoro/kokoro_model_transforms.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines/kokoro/kokoro_split.cpp
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/pipelines/kokoro/kokoro_utils.cpp
            # model builder (test utility)
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/tests/functional/behavior/npuw/test_engine/models/model_builder.cpp
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

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)

target_sources(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/npuw/llm_compiled_model_graph_options_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/npuw/partitioning_options_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/npuw/npuw_options_smoke_test.cpp
)
