# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  bf16_convert_saturation.cpp
  causal_mask_preprocess.cpp
  concat_sdp.cpp
  conv_concat.cpp
  convert_fq_rnn_to_quantized_rnn.cpp
  convert_range.cpp
  conv_maxpool_activ.cpp
  conv_sum_broadcast.cpp
  conv_u8_fuse_sum_i8.cpp
  conv_u8s8f32.cpp
  conv_with_zero_point_fuse.cpp
  fq_caching.cpp
  fq_fused_with_ss.cpp
  fq_layer_dq_bias.cpp
  fuse_conv_fq_with_shared_constants.cpp
  fuse_reshape_transpose_to_sdpa.cpp
  gather_add_avgpool.cpp
  interaction.cpp
  matmul_quantized_subgraph.cpp
  matmul_weights_decompression.cpp
  memory_sharing_test.cpp
  mha.cpp
  mlp_fusion.cpp
  paged_attn.cpp
  paged_attn_score.cpp
  qkv_proj_fusion.cpp
  subgraph_caching.cpp
  subgraph_serialize.cpp
  subgraph_with_blocked_format.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

get_property(subgraph_x64_sources GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_X64_SOURCES)
list(APPEND subgraph_x64_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_X64_SOURCES "${subgraph_x64_sources}")