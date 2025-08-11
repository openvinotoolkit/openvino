# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(collect_custom_sources out_list)
  set(local_srcs)
  set(custom_dir ${CMAKE_CURRENT_SOURCE_DIR}/custom)
  
  # Custom tests (behavior and extensions)
  list(APPEND local_srcs
    ${custom_dir}/behavior/export_import.cpp
    ${custom_dir}/behavior/ov_executable_network/concurent_release_memory.cpp
    ${custom_dir}/behavior/ov_executable_network/properties.cpp
    ${custom_dir}/behavior/ov_executable_network/release_memory.cpp
    ${custom_dir}/behavior/ov_plugin/properties.cpp)

  if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND local_srcs ${custom_dir}/extension/extension.cpp)
  endif()

  # Custom tests: single layer and subgraph
  if(ARM OR AARCH64)
    # Only explicitly enabled subsets on ARM/AARCH64
    # Explicitly enabled ARM/custom tests (previously globbed)
    set(TMP_LIST_OF_TEST_CLASSES
      ${custom_dir}/single_layer_tests/classes/activation.cpp
      ${custom_dir}/single_layer_tests/classes/activation.hpp
      ${custom_dir}/single_layer_tests/classes/bitwise_shift.cpp
      ${custom_dir}/single_layer_tests/classes/bitwise_shift.hpp
      ${custom_dir}/single_layer_tests/classes/col2im.cpp
      ${custom_dir}/single_layer_tests/classes/col2im.hpp
      ${custom_dir}/single_layer_tests/classes/comparison.cpp
      ${custom_dir}/single_layer_tests/classes/comparison.hpp
      ${custom_dir}/single_layer_tests/classes/conversion.cpp
      ${custom_dir}/single_layer_tests/classes/conversion.hpp
      ${custom_dir}/single_layer_tests/classes/convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/classes/convolution_backprop_data.hpp
      ${custom_dir}/single_layer_tests/classes/convolution.cpp
      ${custom_dir}/single_layer_tests/classes/convolution.hpp
      ${custom_dir}/single_layer_tests/classes/eltwise.cpp
      ${custom_dir}/single_layer_tests/classes/eltwise.hpp
      ${custom_dir}/single_layer_tests/classes/extremum.cpp
      ${custom_dir}/single_layer_tests/classes/extremum.hpp
      ${custom_dir}/single_layer_tests/classes/interpolate.cpp
      ${custom_dir}/single_layer_tests/classes/interpolate.hpp
      ${custom_dir}/single_layer_tests/classes/logical.cpp
      ${custom_dir}/single_layer_tests/classes/logical.hpp
      ${custom_dir}/single_layer_tests/classes/matmul.cpp
      ${custom_dir}/single_layer_tests/classes/matmul.hpp
      ${custom_dir}/single_layer_tests/classes/mvn.cpp
      ${custom_dir}/single_layer_tests/classes/mvn.hpp
      ${custom_dir}/single_layer_tests/classes/pooling.cpp
      ${custom_dir}/single_layer_tests/classes/pooling.hpp
      ${custom_dir}/single_layer_tests/classes/random_uniform.cpp
      ${custom_dir}/single_layer_tests/classes/random_uniform.hpp
      ${custom_dir}/single_layer_tests/classes/reduce.cpp
      ${custom_dir}/single_layer_tests/classes/reduce.hpp
      ${custom_dir}/single_layer_tests/classes/rms_norm.cpp
      ${custom_dir}/single_layer_tests/classes/rms_norm.hpp
      ${custom_dir}/single_layer_tests/classes/scaled_attn.cpp
      ${custom_dir}/single_layer_tests/classes/scaled_attn.hpp
      ${custom_dir}/single_layer_tests/classes/segment_max.cpp
      ${custom_dir}/single_layer_tests/classes/segment_max.hpp
      ${custom_dir}/single_layer_tests/classes/softmax.cpp
      ${custom_dir}/single_layer_tests/classes/softmax.hpp
      ${custom_dir}/single_layer_tests/classes/sparse_fill_empty_rows.cpp
      ${custom_dir}/single_layer_tests/classes/sparse_fill_empty_rows.hpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_pack.cpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_pack.hpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_unpack.cpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_unpack.hpp
      ${custom_dir}/single_layer_tests/classes/transpose.cpp
      ${custom_dir}/single_layer_tests/classes/transpose.hpp)

    set(TMP_LIST_OF_COMMON_TEST_INSTANCES
      ${custom_dir}/single_layer_tests/instances/common/activation.cpp
      ${custom_dir}/single_layer_tests/instances/common/bitwise_shift.cpp
      ${custom_dir}/single_layer_tests/instances/common/col2im.cpp
      ${custom_dir}/single_layer_tests/instances/common/comparison.cpp
      ${custom_dir}/single_layer_tests/instances/common/conversion.cpp
      ${custom_dir}/single_layer_tests/instances/common/convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/instances/common/convolution.cpp
      ${custom_dir}/single_layer_tests/instances/common/eltwise.cpp
      ${custom_dir}/single_layer_tests/instances/common/extremum.cpp
      ${custom_dir}/single_layer_tests/instances/common/interpolate.cpp
      ${custom_dir}/single_layer_tests/instances/common/inverse.cpp
      ${custom_dir}/single_layer_tests/instances/common/logical.cpp
      ${custom_dir}/single_layer_tests/instances/common/matmul.cpp
      ${custom_dir}/single_layer_tests/instances/common/multinomial.cpp
      ${custom_dir}/single_layer_tests/instances/common/mvn.cpp
      ${custom_dir}/single_layer_tests/instances/common/nms_rotated.cpp
      ${custom_dir}/single_layer_tests/instances/common/pooling.cpp
      ${custom_dir}/single_layer_tests/instances/common/random_uniform.cpp
      ${custom_dir}/single_layer_tests/instances/common/reduce.cpp
      ${custom_dir}/single_layer_tests/instances/common/scaled_attn.cpp
      ${custom_dir}/single_layer_tests/instances/common/segment_max.cpp
      ${custom_dir}/single_layer_tests/instances/common/softmax.cpp
      ${custom_dir}/single_layer_tests/instances/common/sparse_fill_empty_rows.cpp
      ${custom_dir}/single_layer_tests/instances/common/string_tensor_pack.cpp
      ${custom_dir}/single_layer_tests/instances/common/string_tensor_unpack.cpp
      ${custom_dir}/single_layer_tests/instances/common/transpose.cpp)

    set(TMP_LIST_OF_ARM_TEST_INSTANCES
      ${custom_dir}/single_layer_tests/instances/arm/conversion.cpp
      ${custom_dir}/single_layer_tests/instances/arm/convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/instances/arm/eltwise.cpp
      ${custom_dir}/single_layer_tests/instances/arm/interpolate.cpp
      ${custom_dir}/single_layer_tests/instances/arm/matmul.cpp
      ${custom_dir}/single_layer_tests/instances/arm/pooling.cpp
      ${custom_dir}/single_layer_tests/instances/arm/reduce.cpp)

    set(TMP_LIST_OF_ARM_SUBGRAPH_TESTS
      ${custom_dir}/subgraph_tests/src/arm/concat_sdp.cpp
      ${custom_dir}/subgraph_tests/src/arm/convert_group_conv.cpp
      ${custom_dir}/subgraph_tests/src/arm/convert_group_conv1d.cpp
      ${custom_dir}/subgraph_tests/src/arm/convert_reduce_multi_axis.cpp
      ${custom_dir}/subgraph_tests/src/arm/deconv_multiple_output_edges.cpp
      ${custom_dir}/subgraph_tests/src/arm/eltwise_chain.cpp
      ${custom_dir}/subgraph_tests/src/arm/fuse_eltwise_convert.cpp
      ${custom_dir}/subgraph_tests/src/arm/matmul_weights_decompression.cpp
      ${custom_dir}/subgraph_tests/src/arm/reorder_deconv_nhwc.cpp)

    set(TMP_LIST_OF_COMMON_SUBGRAPH_TESTS
      ${custom_dir}/subgraph_tests/src/common/add_convert_to_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/align_matmul_input_ranks.cpp
      ${custom_dir}/subgraph_tests/src/common/any_layout.cpp
      ${custom_dir}/subgraph_tests/src/common/broadcast_eltwise.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_const_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_conv_sum_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_multiple_query_sdp.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_reorder_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_reshape_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_sdp.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_transpose_sdp_transpose.cpp
      ${custom_dir}/subgraph_tests/src/common/conv3d_reshape.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_dw_conv.cpp
      ${custom_dir}/subgraph_tests/src/common/convert_bool_math.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_maxpool_activ.cpp
      ${custom_dir}/subgraph_tests/src/common/convs_and_sums.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_insert_convert_i64.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_scalar.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_string.cpp
      ${custom_dir}/subgraph_tests/src/common/denormal_check.cpp
      ${custom_dir}/subgraph_tests/src/common/eltwise_caching.cpp
      ${custom_dir}/subgraph_tests/src/common/eltwise_chain.cpp
      ${custom_dir}/subgraph_tests/src/common/fullyconnected_strided_inputs_outputs.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_muladd_ewsimple.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_non0_output_port.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_scaleshift_and_fakequantize.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_split_concat_pair_to_interpolate.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/index_add_scatter_elements_update.cpp
      ${custom_dir}/subgraph_tests/src/common/init_state_inplace_conflicts.cpp
      ${custom_dir}/subgraph_tests/src/common/inplace_edge.cpp
      ${custom_dir}/subgraph_tests/src/common/inplace_resolve_io.cpp
      ${custom_dir}/subgraph_tests/src/common/input_noreorder_eltwise_bf16.cpp
      ${custom_dir}/subgraph_tests/src/common/input_output_tensor_reuse.cpp
      ${custom_dir}/subgraph_tests/src/common/input_tensor_roi.cpp
      ${custom_dir}/subgraph_tests/src/common/lora_pattern.cpp
      ${custom_dir}/subgraph_tests/src/common/matmul_decompress_convert.cpp
      ${custom_dir}/subgraph_tests/src/common/matmul_strided_inputs_outputs.cpp
      ${custom_dir}/subgraph_tests/src/common/merge_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/ngram.cpp
      ${custom_dir}/subgraph_tests/src/common/not_fused_conv_simple_op.cpp
      ${custom_dir}/subgraph_tests/src/common/read_value_assign.cpp
      ${custom_dir}/subgraph_tests/src/common/remove_convert.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_chain.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_fc.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_permute_conv_permute_reshape_act.cpp
      ${custom_dir}/subgraph_tests/src/common/sdpa_group_beam_search.cpp
      ${custom_dir}/subgraph_tests/src/common/seq_native_order.cpp
      ${custom_dir}/subgraph_tests/src/common/shape_infer_subgraph.cpp
      ${custom_dir}/subgraph_tests/src/common/shapeof_any_layout.cpp
      ${custom_dir}/subgraph_tests/src/common/split_concat_add.cpp
      ${custom_dir}/subgraph_tests/src/common/split_matmul_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/stateful_init_graph.cpp
      ${custom_dir}/subgraph_tests/src/common/static_zero_dims.cpp
      ${custom_dir}/subgraph_tests/src/common/strided_slice_zero_dims.cpp
      ${custom_dir}/subgraph_tests/src/common/tile_with_two_output_edges.cpp
      ${custom_dir}/subgraph_tests/src/common/undefined_et.cpp)

    set(TMP_LIST_OF_SUBGRAPH_TEST_CLASSES
      ${custom_dir}/subgraph_tests/src/classes/concat_sdp.cpp
      ${custom_dir}/subgraph_tests/src/classes/concat_sdp.hpp
      ${custom_dir}/subgraph_tests/src/classes/conv_concat.cpp
      ${custom_dir}/subgraph_tests/src/classes/conv_concat.hpp
      ${custom_dir}/subgraph_tests/src/classes/conv_maxpool_activ.cpp
      ${custom_dir}/subgraph_tests/src/classes/conv_maxpool_activ.hpp
      ${custom_dir}/subgraph_tests/src/classes/eltwise_chain.cpp
      ${custom_dir}/subgraph_tests/src/classes/fuse_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/classes/fuse_transpose_reorder.hpp
      ${custom_dir}/subgraph_tests/src/classes/matmul_weights_decompression.cpp
      ${custom_dir}/subgraph_tests/src/classes/matmul_weights_decompression.hpp
      ${custom_dir}/subgraph_tests/src/classes/undefined_et.cpp)

    list(APPEND local_srcs
      ${TMP_LIST_OF_TEST_CLASSES}
      ${TMP_LIST_OF_COMMON_TEST_INSTANCES}
      ${TMP_LIST_OF_ARM_TEST_INSTANCES}
      ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS}
      ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS}
      ${TMP_LIST_OF_SUBGRAPH_TEST_CLASSES})
  else()
    # Full set for other architectures
    # single_layer_tests top-level
    list(APPEND local_srcs
      ${custom_dir}/single_layer_tests/adaptive_pooling.cpp
      ${custom_dir}/single_layer_tests/augru_cell.cpp
      ${custom_dir}/single_layer_tests/augru_sequence.cpp
      ${custom_dir}/single_layer_tests/batch_to_space.cpp
      ${custom_dir}/single_layer_tests/broadcast.cpp
      ${custom_dir}/single_layer_tests/bucketize.cpp
      ${custom_dir}/single_layer_tests/concat.cpp
      ${custom_dir}/single_layer_tests/convert_to_plugin_specific_node.cpp
      ${custom_dir}/single_layer_tests/ctc_greedy_decoder.cpp
      ${custom_dir}/single_layer_tests/ctc_greedy_decoder_seq_len.cpp
      ${custom_dir}/single_layer_tests/ctc_loss.cpp
      ${custom_dir}/single_layer_tests/cum_sum.cpp
      ${custom_dir}/single_layer_tests/custom_op_internal_dyn.cpp
      ${custom_dir}/single_layer_tests/deformable_convolution.cpp
      ${custom_dir}/single_layer_tests/depth_to_space.cpp
      ${custom_dir}/single_layer_tests/detection_output.cpp
      ${custom_dir}/single_layer_tests/embedding_bag_offsets.cpp
      ${custom_dir}/single_layer_tests/embedding_bag_offsets_sum.cpp
      ${custom_dir}/single_layer_tests/embedding_bag_packed.cpp
      ${custom_dir}/single_layer_tests/embedding_bag_packed_sum.cpp
      ${custom_dir}/single_layer_tests/embedding_segments_sum.cpp
      ${custom_dir}/single_layer_tests/extract_image_patches.cpp
      ${custom_dir}/single_layer_tests/eye.cpp
      ${custom_dir}/single_layer_tests/fake_quantize.cpp
      ${custom_dir}/single_layer_tests/gather.cpp
      ${custom_dir}/single_layer_tests/gather_elements.cpp
      ${custom_dir}/single_layer_tests/gather_nd.cpp
      ${custom_dir}/single_layer_tests/gather_tree.cpp
      ${custom_dir}/single_layer_tests/grid_sample.cpp
      ${custom_dir}/single_layer_tests/grn.cpp
      ${custom_dir}/single_layer_tests/group_convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/group_convolution.cpp
      ${custom_dir}/single_layer_tests/gru_cell.cpp
      ${custom_dir}/single_layer_tests/gru_sequence.cpp
      ${custom_dir}/single_layer_tests/log_softmax.cpp
      ${custom_dir}/single_layer_tests/loop.cpp
      ${custom_dir}/single_layer_tests/lrn.cpp
      ${custom_dir}/single_layer_tests/lstm_cell.cpp
      ${custom_dir}/single_layer_tests/lstm_sequence.cpp
      ${custom_dir}/single_layer_tests/matmul_sparse.cpp
      ${custom_dir}/single_layer_tests/non_max_suppression.cpp
      ${custom_dir}/single_layer_tests/nonzero.cpp
      ${custom_dir}/single_layer_tests/normalize.cpp
      ${custom_dir}/single_layer_tests/one_hot.cpp
      ${custom_dir}/single_layer_tests/pad.cpp
      ${custom_dir}/single_layer_tests/prior_box_clustered.cpp
      ${custom_dir}/single_layer_tests/prior_box.cpp
      ${custom_dir}/single_layer_tests/proposal.cpp
      ${custom_dir}/single_layer_tests/psroi_pooling.cpp
      ${custom_dir}/single_layer_tests/rdft.cpp
      ${custom_dir}/single_layer_tests/region_yolo.cpp
      ${custom_dir}/single_layer_tests/reorg_yolo.cpp
      ${custom_dir}/single_layer_tests/reverse_sequence.cpp
      ${custom_dir}/single_layer_tests/rnn_cell.cpp
      ${custom_dir}/single_layer_tests/rnn_sequence.cpp
      ${custom_dir}/single_layer_tests/roialign.cpp
      ${custom_dir}/single_layer_tests/roi_pooling.cpp
      ${custom_dir}/single_layer_tests/roll.cpp
      ${custom_dir}/single_layer_tests/scatter_elements_update.cpp
      ${custom_dir}/single_layer_tests/scatter_ND_update.cpp
      ${custom_dir}/single_layer_tests/scatter_update.cpp
      ${custom_dir}/single_layer_tests/select.cpp
      ${custom_dir}/single_layer_tests/shapeof.cpp
      ${custom_dir}/single_layer_tests/shape_ops.cpp
      ${custom_dir}/single_layer_tests/shuffle_channels.cpp
      ${custom_dir}/single_layer_tests/slice.cpp
      ${custom_dir}/single_layer_tests/slice_scatter.cpp
      ${custom_dir}/single_layer_tests/space_to_batch.cpp
      ${custom_dir}/single_layer_tests/space_to_depth.cpp
      ${custom_dir}/single_layer_tests/split.cpp
      ${custom_dir}/single_layer_tests/strided_slice.cpp
      ${custom_dir}/single_layer_tests/tensor_iterator.cpp
      ${custom_dir}/single_layer_tests/tile.cpp
      ${custom_dir}/single_layer_tests/topk.cpp
      ${custom_dir}/single_layer_tests/unique.cpp
      ${custom_dir}/single_layer_tests/variadic_split.cpp)

    # single_layer_tests classes (definitions used by instances)
    list(APPEND local_srcs
      ${custom_dir}/single_layer_tests/classes/activation.cpp
      ${custom_dir}/single_layer_tests/classes/activation.hpp
      ${custom_dir}/single_layer_tests/classes/bitwise_shift.cpp
      ${custom_dir}/single_layer_tests/classes/bitwise_shift.hpp
      ${custom_dir}/single_layer_tests/classes/col2im.cpp
      ${custom_dir}/single_layer_tests/classes/col2im.hpp
      ${custom_dir}/single_layer_tests/classes/comparison.cpp
      ${custom_dir}/single_layer_tests/classes/comparison.hpp
      ${custom_dir}/single_layer_tests/classes/conversion.cpp
      ${custom_dir}/single_layer_tests/classes/conversion.hpp
      ${custom_dir}/single_layer_tests/classes/convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/classes/convolution_backprop_data.hpp
      ${custom_dir}/single_layer_tests/classes/convolution.cpp
      ${custom_dir}/single_layer_tests/classes/convolution.hpp
      ${custom_dir}/single_layer_tests/classes/eltwise.cpp
      ${custom_dir}/single_layer_tests/classes/eltwise.hpp
      ${custom_dir}/single_layer_tests/classes/extremum.cpp
      ${custom_dir}/single_layer_tests/classes/extremum.hpp
      ${custom_dir}/single_layer_tests/classes/interpolate.cpp
      ${custom_dir}/single_layer_tests/classes/interpolate.hpp
      ${custom_dir}/single_layer_tests/classes/logical.cpp
      ${custom_dir}/single_layer_tests/classes/logical.hpp
      ${custom_dir}/single_layer_tests/classes/matmul.cpp
      ${custom_dir}/single_layer_tests/classes/matmul.hpp
      ${custom_dir}/single_layer_tests/classes/mvn.cpp
      ${custom_dir}/single_layer_tests/classes/mvn.hpp
      ${custom_dir}/single_layer_tests/classes/pooling.cpp
      ${custom_dir}/single_layer_tests/classes/pooling.hpp
      ${custom_dir}/single_layer_tests/classes/random_uniform.cpp
      ${custom_dir}/single_layer_tests/classes/random_uniform.hpp
      ${custom_dir}/single_layer_tests/classes/reduce.cpp
      ${custom_dir}/single_layer_tests/classes/reduce.hpp
      ${custom_dir}/single_layer_tests/classes/rms_norm.cpp
      ${custom_dir}/single_layer_tests/classes/rms_norm.hpp
      ${custom_dir}/single_layer_tests/classes/scaled_attn.cpp
      ${custom_dir}/single_layer_tests/classes/scaled_attn.hpp
      ${custom_dir}/single_layer_tests/classes/segment_max.cpp
      ${custom_dir}/single_layer_tests/classes/segment_max.hpp
      ${custom_dir}/single_layer_tests/classes/softmax.cpp
      ${custom_dir}/single_layer_tests/classes/softmax.hpp
      ${custom_dir}/single_layer_tests/classes/sparse_fill_empty_rows.cpp
      ${custom_dir}/single_layer_tests/classes/sparse_fill_empty_rows.hpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_pack.cpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_pack.hpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_unpack.cpp
      ${custom_dir}/single_layer_tests/classes/string_tensor_unpack.hpp
      ${custom_dir}/single_layer_tests/classes/transpose.cpp
      ${custom_dir}/single_layer_tests/classes/transpose.hpp)

    # single_layer_tests instances (common + arch)
    list(APPEND local_srcs
      ${custom_dir}/single_layer_tests/instances/common/activation.cpp
      ${custom_dir}/single_layer_tests/instances/common/bitwise_shift.cpp
      ${custom_dir}/single_layer_tests/instances/common/col2im.cpp
      ${custom_dir}/single_layer_tests/instances/common/comparison.cpp
      ${custom_dir}/single_layer_tests/instances/common/conversion.cpp
      ${custom_dir}/single_layer_tests/instances/common/convolution_backprop_data.cpp
      ${custom_dir}/single_layer_tests/instances/common/convolution.cpp
      ${custom_dir}/single_layer_tests/instances/common/eltwise.cpp
      ${custom_dir}/single_layer_tests/instances/common/extremum.cpp
      ${custom_dir}/single_layer_tests/instances/common/interpolate.cpp
      ${custom_dir}/single_layer_tests/instances/common/inverse.cpp
      ${custom_dir}/single_layer_tests/instances/common/logical.cpp
      ${custom_dir}/single_layer_tests/instances/common/matmul.cpp
      ${custom_dir}/single_layer_tests/instances/common/multinomial.cpp
      ${custom_dir}/single_layer_tests/instances/common/mvn.cpp
      ${custom_dir}/single_layer_tests/instances/common/nms_rotated.cpp
      ${custom_dir}/single_layer_tests/instances/common/pooling.cpp
      ${custom_dir}/single_layer_tests/instances/common/random_uniform.cpp
      ${custom_dir}/single_layer_tests/instances/common/reduce.cpp
      ${custom_dir}/single_layer_tests/instances/common/scaled_attn.cpp
      ${custom_dir}/single_layer_tests/instances/common/segment_max.cpp
      ${custom_dir}/single_layer_tests/instances/common/softmax.cpp
      ${custom_dir}/single_layer_tests/instances/common/sparse_fill_empty_rows.cpp
      ${custom_dir}/single_layer_tests/instances/common/string_tensor_pack.cpp
      ${custom_dir}/single_layer_tests/instances/common/string_tensor_unpack.cpp
      ${custom_dir}/single_layer_tests/instances/common/transpose.cpp)

    if(X86 OR X86_64)
      list(APPEND local_srcs
        ${custom_dir}/single_layer_tests/instances/x64/activation.cpp
        ${custom_dir}/single_layer_tests/instances/x64/col2im.cpp
        ${custom_dir}/single_layer_tests/instances/x64/conversion.cpp
        ${custom_dir}/single_layer_tests/instances/x64/convolution_backprop_data.cpp
        ${custom_dir}/single_layer_tests/instances/x64/convolution.cpp
        ${custom_dir}/single_layer_tests/instances/x64/eltwise.cpp
        ${custom_dir}/single_layer_tests/instances/x64/interpolate.cpp
        ${custom_dir}/single_layer_tests/instances/x64/matmul.cpp
        ${custom_dir}/single_layer_tests/instances/x64/mvn.cpp
        ${custom_dir}/single_layer_tests/instances/x64/pooling.cpp
        ${custom_dir}/single_layer_tests/instances/x64/random_uniform.cpp
        ${custom_dir}/single_layer_tests/instances/x64/reduce.cpp
        ${custom_dir}/single_layer_tests/instances/x64/rms_norm.cpp
        ${custom_dir}/single_layer_tests/instances/x64/softmax.cpp
        ${custom_dir}/single_layer_tests/instances/x64/transpose.cpp)
    endif()

    if(RISCV64)
      list(APPEND local_srcs
        ${custom_dir}/single_layer_tests/instances/riscv64/eltwise.cpp)
      if(OV_CPU_WITH_SHL)
        list(APPEND local_srcs
          ${custom_dir}/single_layer_tests/instances/riscv64/shl/matmul.cpp)
      endif()
    endif()

    # subgraph tests
    list(APPEND local_srcs
      ${custom_dir}/subgraph_tests/src/disable_gathercompressed_quantized_model.cpp
      ${custom_dir}/subgraph_tests/src/quantized_matmuls_with_shared_weights.cpp
      ${custom_dir}/subgraph_tests/src/subgraph_select_pd.cpp)

    # Common subgraph tests list (redefining since it's used in multiple places)
    set(TMP_LIST_OF_COMMON_SUBGRAPH_TESTS
      ${custom_dir}/subgraph_tests/src/common/add_convert_to_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/align_matmul_input_ranks.cpp
      ${custom_dir}/subgraph_tests/src/common/any_layout.cpp
      ${custom_dir}/subgraph_tests/src/common/broadcast_eltwise.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_const_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_conv_sum_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_multiple_query_sdp.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_reorder_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_reshape_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_sdp.cpp
      ${custom_dir}/subgraph_tests/src/common/concat_transpose_sdp_transpose.cpp
      ${custom_dir}/subgraph_tests/src/common/conv3d_reshape.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_dw_conv.cpp
      ${custom_dir}/subgraph_tests/src/common/convert_bool_math.cpp
      ${custom_dir}/subgraph_tests/src/common/conv_maxpool_activ.cpp
      ${custom_dir}/subgraph_tests/src/common/convs_and_sums.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_insert_convert_i64.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_scalar.cpp
      ${custom_dir}/subgraph_tests/src/common/custom_op_string.cpp
      ${custom_dir}/subgraph_tests/src/common/denormal_check.cpp
      ${custom_dir}/subgraph_tests/src/common/eltwise_caching.cpp
      ${custom_dir}/subgraph_tests/src/common/eltwise_chain.cpp
      ${custom_dir}/subgraph_tests/src/common/fullyconnected_strided_inputs_outputs.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_muladd_ewsimple.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_non0_output_port.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_scaleshift_and_fakequantize.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_split_concat_pair_to_interpolate.cpp
      ${custom_dir}/subgraph_tests/src/common/fuse_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/index_add_scatter_elements_update.cpp
      ${custom_dir}/subgraph_tests/src/common/init_state_inplace_conflicts.cpp
      ${custom_dir}/subgraph_tests/src/common/inplace_edge.cpp
      ${custom_dir}/subgraph_tests/src/common/inplace_resolve_io.cpp
      ${custom_dir}/subgraph_tests/src/common/input_noreorder_eltwise_bf16.cpp
      ${custom_dir}/subgraph_tests/src/common/input_output_tensor_reuse.cpp
      ${custom_dir}/subgraph_tests/src/common/input_tensor_roi.cpp
      ${custom_dir}/subgraph_tests/src/common/lora_pattern.cpp
      ${custom_dir}/subgraph_tests/src/common/matmul_decompress_convert.cpp
      ${custom_dir}/subgraph_tests/src/common/matmul_strided_inputs_outputs.cpp
      ${custom_dir}/subgraph_tests/src/common/merge_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/common/ngram.cpp
      ${custom_dir}/subgraph_tests/src/common/not_fused_conv_simple_op.cpp
      ${custom_dir}/subgraph_tests/src/common/read_value_assign.cpp
      ${custom_dir}/subgraph_tests/src/common/remove_convert.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_chain.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_fc.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_inplace.cpp
      ${custom_dir}/subgraph_tests/src/common/reshape_permute_conv_permute_reshape_act.cpp
      ${custom_dir}/subgraph_tests/src/common/sdpa_group_beam_search.cpp
      ${custom_dir}/subgraph_tests/src/common/seq_native_order.cpp
      ${custom_dir}/subgraph_tests/src/common/shape_infer_subgraph.cpp
      ${custom_dir}/subgraph_tests/src/common/shapeof_any_layout.cpp
      ${custom_dir}/subgraph_tests/src/common/split_concat_add.cpp
      ${custom_dir}/subgraph_tests/src/common/split_matmul_concat.cpp
      ${custom_dir}/subgraph_tests/src/common/stateful_init_graph.cpp
      ${custom_dir}/subgraph_tests/src/common/static_zero_dims.cpp
      ${custom_dir}/subgraph_tests/src/common/strided_slice_zero_dims.cpp
      ${custom_dir}/subgraph_tests/src/common/tile_with_two_output_edges.cpp
      ${custom_dir}/subgraph_tests/src/common/undefined_et.cpp)

    list(APPEND local_srcs ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS})

    # subgraph tests classes (common definitions used by subgraph tests where applicable)
    list(APPEND local_srcs
      ${custom_dir}/subgraph_tests/src/classes/concat_sdp.cpp
      ${custom_dir}/subgraph_tests/src/classes/concat_sdp.hpp
      ${custom_dir}/subgraph_tests/src/classes/conv_concat.cpp
      ${custom_dir}/subgraph_tests/src/classes/conv_concat.hpp
      ${custom_dir}/subgraph_tests/src/classes/conv_maxpool_activ.cpp
      ${custom_dir}/subgraph_tests/src/classes/conv_maxpool_activ.hpp
      ${custom_dir}/subgraph_tests/src/classes/eltwise_chain.cpp
      ${custom_dir}/subgraph_tests/src/classes/fuse_transpose_reorder.cpp
      ${custom_dir}/subgraph_tests/src/classes/fuse_transpose_reorder.hpp
      ${custom_dir}/subgraph_tests/src/classes/matmul_weights_decompression.cpp
      ${custom_dir}/subgraph_tests/src/classes/matmul_weights_decompression.hpp
      ${custom_dir}/subgraph_tests/src/classes/undefined_et.cpp)

    if(X86 OR X86_64)
      list(APPEND local_srcs
        ${custom_dir}/subgraph_tests/src/x64/bf16_convert_saturation.cpp
        ${custom_dir}/subgraph_tests/src/x64/causal_mask_preprocess.cpp
        ${custom_dir}/subgraph_tests/src/x64/concat_sdp.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_concat.cpp
        ${custom_dir}/subgraph_tests/src/x64/convert_fq_rnn_to_quantized_rnn.cpp
        ${custom_dir}/subgraph_tests/src/x64/convert_range.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_maxpool_activ.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_sum_broadcast.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_u8_fuse_sum_i8.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_u8s8f32.cpp
        ${custom_dir}/subgraph_tests/src/x64/conv_with_zero_point_fuse.cpp
        ${custom_dir}/subgraph_tests/src/x64/fq_caching.cpp
        ${custom_dir}/subgraph_tests/src/x64/fq_fused_with_ss.cpp
        ${custom_dir}/subgraph_tests/src/x64/fq_layer_dq_bias.cpp
        ${custom_dir}/subgraph_tests/src/x64/fuse_conv_fq_with_shared_constants.cpp
        ${custom_dir}/subgraph_tests/src/x64/fuse_reshape_transpose_to_sdpa.cpp
        ${custom_dir}/subgraph_tests/src/x64/gather_add_avgpool.cpp
        ${custom_dir}/subgraph_tests/src/x64/interaction.cpp
        ${custom_dir}/subgraph_tests/src/x64/matmul_quantized_subgraph.cpp
        ${custom_dir}/subgraph_tests/src/x64/matmul_weights_decompression.cpp
        ${custom_dir}/subgraph_tests/src/x64/memory_sharing_test.cpp
        ${custom_dir}/subgraph_tests/src/x64/mha.cpp
        ${custom_dir}/subgraph_tests/src/x64/mlp_fusion.cpp
        ${custom_dir}/subgraph_tests/src/x64/paged_attn.cpp
        ${custom_dir}/subgraph_tests/src/x64/paged_attn_score.cpp
        ${custom_dir}/subgraph_tests/src/x64/qkv_proj_fusion.cpp
        ${custom_dir}/subgraph_tests/src/x64/subgraph_caching.cpp
        ${custom_dir}/subgraph_tests/src/x64/subgraph_serialize.cpp
        ${custom_dir}/subgraph_tests/src/x64/subgraph_with_blocked_format.cpp)
    endif()
  endif()

  # Return the collected sources
  set(${out_list} "${local_srcs}" PARENT_SCOPE)
endfunction()