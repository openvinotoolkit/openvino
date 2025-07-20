# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(x64_SRC
    bf16_convert_saturation.cpp
    causal_mask_preprocess.cpp
    concat_sdp.cpp
    conv_concat.cpp
    conv_maxpool_activ.cpp
    conv_sum_broadcast.cpp
    conv_u8_fuse_sum_i8.cpp
    conv_u8s8f32.cpp
    conv_with_zero_point_fuse.cpp
    convert_fq_rnn_to_quantized_rnn.cpp
    convert_range.cpp
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

# Convert to full paths
foreach(src_file ${x64_SRC})
    list(APPEND x64_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${x64_SRC_FULL})
