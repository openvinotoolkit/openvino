# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
  # exclude every other test file inside directory (we pass explicit OBJECT_FILES)
  set(EXCLUDED_SOURCE_PATHS_FOR_TEST ${TEST_DIR})

  # list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp)

  if(X86_64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
  elseif(ARM OR AARCH64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
  elseif(RISCV64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
  endif()

  # Explicit lists for directories
  set(SLT_TOP_LEVEL
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

  set(SUBGRAPH_SRC_TOP_LEVEL
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/disable_gathercompressed_quantized_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/quantized_matmuls_with_shared_weights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/subgraph_select_pd.cpp)

  set(SUBGRAPH_COMMON
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

  set(SUBGRAPH_X64
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

  set(SUBGRAPH_ARM
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/concat_sdp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_group_conv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_group_conv1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/convert_reduce_multi_axis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/deconv_multiple_output_edges.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/eltwise_chain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/fuse_eltwise_convert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/matmul_weights_decompression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/reorder_deconv_nhwc.cpp)

  # 1) Create target per direct .cpp under TEST_DIR
  set(LIST_OF_TEST_FILES)
  if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
    set(LIST_OF_TEST_FILES ${SLT_TOP_LEVEL})
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src")
    set(LIST_OF_TEST_FILES ${SUBGRAPH_SRC_TOP_LEVEL})
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common")
    set(LIST_OF_TEST_FILES ${SUBGRAPH_COMMON})
  endif()

  foreach(TEST_FILE IN LISTS LIST_OF_TEST_FILES)
    get_filename_component(TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_FILE_WE})
    ov_add_test_target(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_FILE}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      ADD_CPPLINT
      LABELS OV CPU)
    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
  endforeach()

  # 2) Class-based targets with instances
  set(LIST_OF_TEST_CLASSES)
  if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
    set(LIST_OF_TEST_CLASSES
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.cpp)
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src")
    set(LIST_OF_TEST_CLASSES
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/eltwise_chain.cpp)
  endif()

  foreach(TEST_CLASS_FILE IN LISTS LIST_OF_TEST_CLASSES)
    get_filename_component(TEST_CLASS ${TEST_CLASS_FILE} NAME_WE)
    get_filename_component(TEST_CLASS_FILE_NAME ${TEST_CLASS_FILE} NAME)
    set(TEST_INSTANCES)
    if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
      set(common_inst ${TEST_DIR}/instances/common/${TEST_CLASS_FILE_NAME})
      if(EXISTS ${common_inst})
        list(APPEND TEST_INSTANCES ${common_inst})
      endif()
      if(X86_64)
        set(arch_inst ${TEST_DIR}/instances/x64/${TEST_CLASS_FILE_NAME})
      elseif(ARM OR AARCH64)
        set(arch_inst ${TEST_DIR}/instances/arm/${TEST_CLASS_FILE_NAME})
      elseif(RISCV64)
        set(arch_inst ${TEST_DIR}/instances/riscv64/${TEST_CLASS_FILE_NAME})
      endif()
      if(DEFINED arch_inst AND EXISTS ${arch_inst})
        list(APPEND TEST_INSTANCES ${arch_inst})
      endif()
    endif()

    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_CLASS})
    ov_add_test_target(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_CLASS_FILE} ${TEST_INSTANCES}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      LABELS OV CPU)
    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
  endforeach()

endfunction()

if(ENABLE_CPU_SPECIFIC_TARGET_PER_TEST)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src ov_cpu_func_subgraph)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common ov_cpu_func_subgraph_common)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests ov_cpu_func_slt)
endif()

# examples of targets:
# - ov_cpu_func_subgraph_mha
# - ov_cpu_func_slt_convolution
