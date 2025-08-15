# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/add_convert_to_reorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/align_matmul_input_ranks.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/any_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_eltwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_const_inplace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_conv_sum_inplace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_multiple_query_sdp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_reorder_inplace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_reshape_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_sdp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_transpose_sdp_transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv3d_reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_dw_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_bool_math.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_maxpool_activ.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convs_and_sums.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_insert_convert_i64.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_scalar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_string.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/denormal_check.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_caching.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_chain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fullyconnected_strided_inputs_outputs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_muladd_ewsimple.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_non0_output_port.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_scaleshift_and_fakequantize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_split_concat_pair_to_interpolate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_transpose_reorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/index_add_scatter_elements_update.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init_state_inplace_conflicts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inplace_edge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inplace_resolve_io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/input_noreorder_eltwise_bf16.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/input_output_tensor_reuse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/input_tensor_roi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lora_pattern.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_decompress_convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_strided_inputs_outputs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/merge_transpose_reorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ngram.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/not_fused_conv_simple_op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/read_value_assign.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/remove_convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_chain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_fc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_inplace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_permute_conv_permute_reshape_act.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_group_beam_search.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/seq_native_order.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/shape_infer_subgraph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/shapeof_any_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/split_concat_add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/split_matmul_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stateful_init_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/static_zero_dims.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/strided_slice_zero_dims.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tile_with_two_output_edges.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/undefined_et.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")