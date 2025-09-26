# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  add_convert_to_reorder.cpp
  align_matmul_input_ranks.cpp
  any_layout.cpp
  broadcast_eltwise.cpp
  concat_const_inplace.cpp
  concat_conv_sum_inplace.cpp
  concat_multiple_query_sdp.cpp
  concat_reorder_inplace.cpp
  concat_reshape_concat.cpp
  concat_sdp.cpp
  concat_transpose_sdp_transpose.cpp
  conv3d_reshape.cpp
  conv_concat.cpp
  conv_dw_conv.cpp
  convert_bool_math.cpp
  conv_maxpool_activ.cpp
  convs_and_sums.cpp
  custom_op_insert_convert_i64.cpp
  custom_op_scalar.cpp
  custom_op_string.cpp
  denormal_check.cpp
  eltwise_caching.cpp
  eltwise_chain.cpp
  fullyconnected_strided_inputs_outputs.cpp
  fuse_muladd_ewsimple.cpp
  fuse_non0_output_port.cpp
  fuse_scaleshift_and_fakequantize.cpp
  fuse_split_concat_pair_to_interpolate.cpp
  fuse_transpose_reorder.cpp
  index_add_scatter_elements_update.cpp
  init_state_inplace_conflicts.cpp
  inplace_edge.cpp
  inplace_resolve_io.cpp
  input_noreorder_eltwise_bf16.cpp
  input_output_tensor_reuse.cpp
  input_tensor_roi.cpp
  lora_pattern.cpp
  matmul_decompress_convert.cpp
  matmul_strided_inputs_outputs.cpp
  merge_transpose_reorder.cpp
  ngram.cpp
  not_fused_conv_simple_op.cpp
  read_value_assign.cpp
  remove_convert.cpp
  reshape_chain.cpp
  reshape_fc.cpp
  reshape_inplace.cpp
  reshape_permute_conv_permute_reshape_act.cpp
  sdpa_group_beam_search.cpp
  seq_native_order.cpp
  shape_infer_subgraph.cpp
  shapeof_any_layout.cpp
  split_concat_add.cpp
  split_matmul_concat.cpp
  stateful_init_graph.cpp
  static_zero_dims.cpp
  strided_slice_zero_dims.cpp
  tile_with_two_output_edges.cpp
  undefined_et.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

get_property(subgraph_common_sources GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_COMMON_SOURCES)
list(APPEND subgraph_common_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_COMMON_SOURCES "${subgraph_common_sources}")