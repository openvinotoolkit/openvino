# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/bf16_convert_saturation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/causal_mask_preprocess.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_sdp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_fq_rnn_to_quantized_rnn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_range.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_maxpool_activ.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_sum_broadcast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_u8_fuse_sum_i8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_u8s8f32.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_with_zero_point_fuse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fq_caching.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fq_fused_with_ss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fq_layer_dq_bias.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_conv_fq_with_shared_constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_reshape_transpose_to_sdpa.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gather_add_avgpool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interaction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_quantized_subgraph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_sharing_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paged_attn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/paged_attn_score.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/qkv_proj_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_caching.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_serialize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_with_blocked_format.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")