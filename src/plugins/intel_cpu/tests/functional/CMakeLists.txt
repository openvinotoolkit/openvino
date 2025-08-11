# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_func_tests)

if(SUGGEST_OVERRIDE_SUPPORTED)
    # xbyak compilation fails
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

# To allow test classes call `mayiuse(gv)`
if(RISCV64)
    set(CPU_ISA_TRAITS_RV64 $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/riscv64/cpu_isa_traits.hpp
                            $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/riscv64/cpu_isa_traits.cpp)
endif()

add_library(cpuUtils STATIC
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.hpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/precision_support.h
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/precision_support.cpp
    ${CPU_ISA_TRAITS_RV64})
set(CPU_UTILS_LINK_LIBRARIES openvino::runtime::dev)
set(CPU_UTILS_INCLUDE_PATHS)
if(OV_CPU_WITH_ACL)
    list(APPEND CPU_UTILS_LINK_LIBRARIES arm_compute::arm_compute)
    list(APPEND CPU_UTILS_INCLUDE_PATHS $<TARGET_PROPERTY:arm_compute::arm_compute,SOURCE_DIR>)
endif()
if(OV_CPU_WITH_DNNL)
    list(APPEND CPU_UTILS_LINK_LIBRARIES dnnl)
    list(APPEND CPU_UTILS_INCLUDE_PATHS
        $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/src
        $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/third_party)
endif()
if(RISCV64)
    list(APPEND CPU_UTILS_LINK_LIBRARIES xbyak_riscv)
endif()
target_link_libraries(cpuUtils PRIVATE ${CPU_UTILS_LINK_LIBRARIES})
target_include_directories(cpuUtils PRIVATE ${CPU_UTILS_INCLUDE_PATHS})

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin openvino_template_extension)
set(LINK_LIBRARIES funcSharedTests cpuUtils openvino::snippets ov_snippets_models)

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom/extension ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/onnx)
endif()

if(NOT RISCV64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64)
elseif(NOT OV_CPU_WITH_SHL)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64/shl)
endif()

if(NOT (X86 OR X86_64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64)
endif()

if(NOT (ARM OR AARCH64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/aarch64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm)
else()
    # temporary disable all custom tests for ARM except explicitly enabled ones
    list(APPEND EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests)

    # Explicitly enabled ARM/custom tests (previously globbed)
    set(TMP_LIST_OF_TEST_CLASSES
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.hpp)

    set(TMP_LIST_OF_COMMON_TEST_INSTANCES
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/bitwise_shift.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/col2im.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/comparison.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/conversion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/extremum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/inverse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/logical.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/multinomial.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/mvn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/nms_rotated.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/random_uniform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/scaled_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/segment_max.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/sparse_fill_empty_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/string_tensor_pack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/string_tensor_unpack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/transpose.cpp)

    set(TMP_LIST_OF_ARM_TEST_INSTANCES
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/conversion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/reduce.cpp)

    set(TMP_LIST_OF_ARM_SUBGRAPH_TESTS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/concat_sdp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_group_conv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_group_conv1d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_reduce_multi_axis.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/deconv_multiple_output_edges.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/eltwise_chain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/fuse_eltwise_convert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/matmul_weights_decompression.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/reorder_deconv_nhwc.cpp)

    set(TMP_LIST_OF_COMMON_SUBGRAPH_TESTS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/add_convert_to_reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/align_matmul_input_ranks.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/any_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/broadcast_eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_const_inplace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_conv_sum_inplace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_multiple_query_sdp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_reorder_inplace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_reshape_concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_sdp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/concat_transpose_sdp_transpose.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/conv3d_reshape.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/conv_concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/conv_dw_conv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/convert_bool_math.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/conv_maxpool_activ.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/convs_and_sums.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/custom_op_insert_convert_i64.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/custom_op_scalar.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/custom_op_string.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/denormal_check.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/eltwise_caching.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/eltwise_chain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fullyconnected_strided_inputs_outputs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fuse_muladd_ewsimple.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fuse_non0_output_port.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fuse_scaleshift_and_fakequantize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fuse_split_concat_pair_to_interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/fuse_transpose_reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/index_add_scatter_elements_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/init_state_inplace_conflicts.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/inplace_edge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/inplace_resolve_io.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/input_noreorder_eltwise_bf16.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/input_output_tensor_reuse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/input_tensor_roi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/lora_pattern.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/matmul_decompress_convert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/matmul_strided_inputs_outputs.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/merge_transpose_reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/ngram.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/not_fused_conv_simple_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/read_value_assign.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/remove_convert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/reshape_chain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/reshape_fc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/reshape_inplace.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/reshape_permute_conv_permute_reshape_act.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/sdpa_group_beam_search.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/seq_native_order.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/shape_infer_subgraph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/shapeof_any_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/split_concat_add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/split_matmul_concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/stateful_init_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/static_zero_dims.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/strided_slice_zero_dims.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/tile_with_two_output_edges.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/undefined_et.cpp)

    set(TMP_LIST_OF_SUBGRAPH_TEST_CLASSES
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/eltwise_chain.cpp)

    list(APPEND TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS
        ${TMP_LIST_OF_TEST_CLASSES}
        ${TMP_LIST_OF_COMMON_TEST_INSTANCES}
        ${TMP_LIST_OF_ARM_TEST_INSTANCES}
        ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS}
        ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS}
        ${TMP_LIST_OF_SUBGRAPH_TEST_CLASSES})
    set(TMP_EXPLICITLY_ENABLED_TESTS "${TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS}")
endif()

# Enumerate all CPU functional test sources explicitly and feed only explicit files to the target
set(CPU_FUNC_TEST_SRCS)

# Utils (common)
list(APPEND CPU_FUNC_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/filter_cpu_info.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/quantization_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/properties_test.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_params.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.hpp)

# Utils (arch-specific)
if(X86_64)
    list(APPEND CPU_FUNC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
    list(APPEND CPU_FUNC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
elseif(RISCV64)
    list(APPEND CPU_FUNC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
endif()

# Shared tests instances (core + SLT + snippets + subgraphs)
list(APPEND CPU_FUNC_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/add_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/disable_lowering_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/duplicate_inputs_outputs_names.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/keep_assign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/nms_transformation_for_last_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/num_inputs_fusing_bin_conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/remove_parameter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/execution_graph_tests/runtime_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/preprocessing/resize_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/preprocessing/yuv_to_grey_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/adaptive_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/batch_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/binary_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/bucketize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/clamp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/comparison.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/convert_color_i420.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/convert_color_nv12.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/ctc_greedy_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/ctc_greedy_decoder_seq_len.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/ctc_loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/cum_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/deformable_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/deformable_psroi_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/detection_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/dft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/embedding_bag_offsets_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/embedding_bag_packed_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/embedding_segments_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/experimental_detectron_detection_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/experimental_detectron_generate_proposals_single_image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/experimental_detectron_prior_grid_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/experimental_detectron_roifeatureextractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/experimental_detectron_topkrois.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/extract_image_patches.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/fake_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gather_elements.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gather_nd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/generate_proposals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/grn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/group_convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/group_normalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gru_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/gru_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/is_inf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/istft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/logical.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/log_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/lrn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/lstm_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/lstm_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/mat_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/matrix_nms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/minimum_maximum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/multiclass_nms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/mvn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/nonzero.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/normalize_l2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/one_hot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/prior_box_clustered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/prior_box.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/psroi_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/rdft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/reduce_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/region_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/result.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/reverse_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/rnn_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/rnn_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/roi_align.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/roi_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/roll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/scatter_elements_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/scatter_ND_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/scatter_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/single_layer_tests/variadic_split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/check_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/codegen_gelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/conv_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/edge_replace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/eltwise_two_results.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/enforce_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/exp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/fake_quantize_decomposition_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/fully_connected.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/gated_mlp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/group_normalization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/max_num_params_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_extracted_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_fma.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_split_dim_m.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_transposed_b.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_with_dyn_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mha_wo_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/mlp_seq.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/precision_propagation_convertion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/three_inputs_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/transpose_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/transpose_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/two_inputs_and_outputs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/broadcast_eltwise_eliminated.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/concat_resize_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/constant_result.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/conv_eltwise_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/convert_pad_to_group_conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/conv_strides_opt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/gather_weights_decompression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/get_output_before_activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/integer_reduce_mean.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/lora_pattern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/matmul_const_transposes_extraction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/matmul_multiply_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/matmul_squeeze_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/memory_LSTMCell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/mul_conv_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/multiple_LSTMCell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/multiply_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/mvn_multiply_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/parameter_result.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/perm_conv_perm_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/quantized_convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/quantized_convolution_batch_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/quantized_group_convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/quantized_group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/quantized_mat_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/range_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/relu_shape_of.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/reshape_squeeze_reshape_relu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/rotary_pos_emb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/shared_matmul_gather_weights_decompression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/shared_matmul_weights_decompression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/simple_if.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/split_concat_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/split_conv_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/stateful_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/transpose_matmul_fusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/subgraph_tests/variadic_split_pad.cpp)

# Shared tests instances arch-specific
if(X86_64)
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/concat_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/concat_with_different_precision_on_children.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/concat_with_neighbors_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/concat_with_split_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/fq_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/fuse_convert_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/fuse_dequantize_to_fq_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/fully_connected_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/groupconvolution_qdq_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/mat_mul_with_optimized_constant_fq.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/mvn_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/output_layers.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/recurrent_cell_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/reduce_min_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/reshape_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64/slice_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/x64/convert.cpp)
elseif(ARM OR AARCH64)
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/aarch64/fully_connected_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/aarch64/mat_mul_transformation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/arm/convert.cpp)
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND CPU_FUNC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/onnx/quantized_models_tests.cpp)
endif()

# Custom tests (behavior and extensions)
list(APPEND CPU_FUNC_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/behavior/export_import.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/behavior/ov_executable_network/concurent_release_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/behavior/ov_executable_network/properties.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/behavior/ov_executable_network/release_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/behavior/ov_plugin/properties.cpp)

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND CPU_FUNC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/custom/extension/extension.cpp)
endif()

# Custom tests: single layer and subgraph
if(ARM OR AARCH64)
    # Only explicitly enabled subsets on ARM/AARCH64
    list(APPEND CPU_FUNC_TEST_SRCS ${TMP_EXPLICITLY_ENABLED_TESTS})
else()
    # Full set for other architectures
    # single_layer_tests top-level
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/adaptive_pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/augru_cell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/augru_sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/batch_to_space.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/broadcast.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/bucketize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/convert_to_plugin_specific_node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/ctc_greedy_decoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/ctc_greedy_decoder_seq_len.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/ctc_loss.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/cum_sum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/custom_op_internal_dyn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/deformable_convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/depth_to_space.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/detection_output.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/embedding_bag_offsets.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/embedding_bag_offsets_sum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/embedding_bag_packed.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/embedding_bag_packed_sum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/embedding_segments_sum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/extract_image_patches.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/eye.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/fake_quantize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gather.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gather_elements.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gather_nd.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gather_tree.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/grid_sample.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/grn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/group_convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/group_convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gru_cell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/gru_sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/log_softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/loop.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/lrn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/lstm_cell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/lstm_sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/matmul_sparse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/non_max_suppression.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/nonzero.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/normalize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/one_hot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/pad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/prior_box_clustered.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/prior_box.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/proposal.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/psroi_pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/rdft.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/region_yolo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/reorg_yolo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/reverse_sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/rnn_cell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/rnn_sequence.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/roialign.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/roi_pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/roll.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/scatter_elements_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/scatter_ND_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/scatter_update.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/select.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/shapeof.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/shape_ops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/shuffle_channels.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/slice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/slice_scatter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/space_to_batch.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/space_to_depth.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/split.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/strided_slice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/tensor_iterator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/tile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/topk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/unique.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/variadic_split.cpp)

    # single_layer_tests classes (definitions used by instances)
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.hpp)

    # single_layer_tests instances (common + arch)
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/activation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/bitwise_shift.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/col2im.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/comparison.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/conversion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/convolution_backprop_data.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/convolution.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/extremum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/interpolate.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/inverse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/logical.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/multinomial.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/mvn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/nms_rotated.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/pooling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/random_uniform.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/reduce.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/scaled_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/segment_max.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/sparse_fill_empty_rows.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/string_tensor_pack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/string_tensor_unpack.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/transpose.cpp)

    if(X86 OR X86_64)
        list(APPEND CPU_FUNC_TEST_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/activation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/col2im.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/conversion.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/convolution_backprop_data.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/convolution.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/eltwise.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/interpolate.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/matmul.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/mvn.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/pooling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/random_uniform.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/reduce.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/rms_norm.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/softmax.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64/transpose.cpp)
    endif()

    if(RISCV64)
        list(APPEND CPU_FUNC_TEST_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64/eltwise.cpp)
        if(OV_CPU_WITH_SHL)
            list(APPEND CPU_FUNC_TEST_SRCS
                ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64/shl/matmul.cpp)
        endif()
    endif()

    # subgraph tests
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/disable_gathercompressed_quantized_model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/quantized_matmuls_with_shared_weights.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/subgraph_select_pd.cpp)

    list(APPEND CPU_FUNC_TEST_SRCS ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS})

    # subgraph tests classes (common definitions used by subgraph tests where applicable)
    list(APPEND CPU_FUNC_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/eltwise_chain.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/fuse_transpose_reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/fuse_transpose_reorder.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/matmul_weights_decompression.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/matmul_weights_decompression.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/undefined_et.cpp)

    if(X86 OR X86_64)
        list(APPEND CPU_FUNC_TEST_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/bf16_convert_saturation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/causal_mask_preprocess.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/concat_sdp.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_concat.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/convert_fq_rnn_to_quantized_rnn.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/convert_range.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_maxpool_activ.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_sum_broadcast.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_u8_fuse_sum_i8.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_u8s8f32.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/conv_with_zero_point_fuse.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/fq_caching.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/fq_fused_with_ss.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/fq_layer_dq_bias.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/fuse_conv_fq_with_shared_constants.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/fuse_reshape_transpose_to_sdpa.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/gather_add_avgpool.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/interaction.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/matmul_quantized_subgraph.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/matmul_weights_decompression.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/memory_sharing_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/mha.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/mlp_fusion.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/paged_attn.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/paged_attn_score.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/qkv_proj_fusion.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/subgraph_caching.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/subgraph_serialize.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64/subgraph_with_blocked_format.cpp)
    endif()
endif()

# Keep IDE structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${CPU_FUNC_TEST_SRCS})

if(NOT X86_64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64)
endif()

# Create the test target with explicit sources only
ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR} # required by helper, but we exclude it entirely below
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS OV CPU
)

# Attach explicit sources to the target directly
target_sources(${TARGET_NAME} PRIVATE ${CPU_FUNC_TEST_SRCS})

ov_set_threading_interface_for(${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)

include(cmake/specific_tests.cmake)
include(cmake/target_per_test.cmake)

# exclude 'ov_cpu_func_tests' from target 'all' if specific tests path was specified
if(DEFINED ENABLE_CPU_SPECIFIC_TESTS_PATH)
    set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
