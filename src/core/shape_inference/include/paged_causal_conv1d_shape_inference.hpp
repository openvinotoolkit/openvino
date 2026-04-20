// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace ov::op::internal {
template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes.size() == 9,
                           "Incorrect number of inputs for PagedCausalConv1D");

    // Per spec:
    // input_embeds: [batch_size_in_tokens, hidden_size] - rank 2
    // conv_state_table: [num_blocks, hidden_size, kernel_size] - rank 3
    // conv_weight: [out_channels, hidden_size/group_size, kernel_size] - rank 3
    // conv_bias: [out_channels] or [0] - rank 1
    // subsequence_begins: [batch_size_in_sequences + 1] - rank 1
    // la_block_indices: [num_blocks] - rank 1
    // la_block_indices_begins: [batch_size_in_sequences + 1] - rank 1
    // processed_tokens: [batch_size_in_sequences] - rank 1
    // cache_interval: [batch_size_in_sequences] - rank 1

    const auto& input_embeds_ps = input_shapes[0];
    const auto& conv_state_table_ps = input_shapes[1];
    const auto& conv_weight_ps = input_shapes[2];
    const auto& conv_bias_ps = input_shapes[3];
    const auto& subsequence_begins_ps = input_shapes[4];
    const auto& la_block_indices_ps = input_shapes[5];
    const auto& la_block_indices_begins_ps = input_shapes[6];
    const auto& processed_tokens_ps = input_shapes[7];
    const auto& cache_interval_ps = input_shapes[8];

    // Rank validation is done by paged_causal_conv1d_input_check in validate_and_infer_types()
    // Here we only check if ranks are static to enable dimension compatibility checks
    const bool embeds_static = input_embeds_ps.rank().is_static();
    const bool state_static = conv_state_table_ps.rank().is_static();
    const bool weight_static = conv_weight_ps.rank().is_static();
    const bool bias_static = conv_bias_ps.rank().is_static();
    const bool subseq_static = subsequence_begins_ps.rank().is_static();
    const bool block_idx_static = la_block_indices_ps.rank().is_static();
    const bool block_begins_static = la_block_indices_begins_ps.rank().is_static();
    const bool proc_tokens_static = processed_tokens_ps.rank().is_static();
    const bool cache_int_static = cache_interval_ps.rank().is_static();

    // hidden_size compatibility: input_embeds[1] == conv_state_table[1]
    if (embeds_static && state_static) {
        const auto input_hidden_size = input_embeds_ps[1];
        const auto state_hidden_size = conv_state_table_ps[1];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_hidden_size.compatible(state_hidden_size),
                               "The hidden_size dimensions of input_embeds and conv_state_table inputs must be "
                               "compatible. Got: input_embeds=",
                               input_hidden_size,
                               ", conv_state_table=",
                               state_hidden_size,
                               ".");
    }

    // kernel_size compatibility: conv_state_table[2] == conv_weight[2]
    if (state_static && weight_static) {
        const auto state_kernel_size = conv_state_table_ps[2];
        const auto weight_kernel_size = conv_weight_ps[2];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               state_kernel_size.compatible(weight_kernel_size),
                               "The kernel_size dimensions of conv_state_table and conv_weight inputs must be "
                               "compatible. Got: conv_state_table=",
                               state_kernel_size,
                               ", conv_weight=",
                               weight_kernel_size,
                               ".");
    }

    // out_channels == hidden_size: conv_weight[0] == input_embeds[1]
    if (embeds_static && weight_static) {
        const auto weight_out_channels = conv_weight_ps[0];
        const auto input_hidden_size = input_embeds_ps[1];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               weight_out_channels.compatible(input_hidden_size),
                               "The out_channels dimension of conv_weight must be compatible with the hidden_size "
                               "dimension of input_embeds. Got: conv_weight=",
                               weight_out_channels,
                               ", input_embeds=",
                               input_hidden_size,
                               ".");
    }

    // conv_bias compatibility: conv_bias[0] == conv_weight[0] (out_channels) or conv_bias[0] == 0
    if (bias_static && weight_static) {
        const auto bias_size = conv_bias_ps[0];
        const auto weight_out_channels = conv_weight_ps[0];
        const auto zero = ov::Dimension(0);

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               bias_size.compatible(weight_out_channels) || bias_size.compatible(zero),
                               "The size of conv_bias must be compatible with the out_channels dimension of "
                               "conv_weight or equal to 0 (no bias). Got: conv_bias=",
                               bias_size,
                               ", conv_weight=",
                               weight_out_channels,
                               ".");
    }

    // num_blocks compatibility: la_block_indices[0] == conv_state_table[0]
    if (state_static && block_idx_static) {
        const auto state_num_blocks = conv_state_table_ps[0];
        const auto indices_num_blocks = la_block_indices_ps[0];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               state_num_blocks.compatible(indices_num_blocks),
                               "The num_blocks dimension of la_block_indices must be compatible with the num_blocks "
                               "dimension of conv_state_table. Got: la_block_indices=",
                               indices_num_blocks,
                               ", conv_state_table=",
                               state_num_blocks,
                               ".");
    }

    // batch_size_in_sequences + 1 compatibility: subsequence_begins[0] == la_block_indices_begins[0]
    if (subseq_static && block_begins_static) {
        const auto subseq_dim = subsequence_begins_ps[0];
        const auto block_begins_dim = la_block_indices_begins_ps[0];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               subseq_dim.compatible(block_begins_dim),
                               "The size of subsequence_begins must be compatible with the size of "
                               "la_block_indices_begins (both should be batch_size_in_sequences + 1). Got: "
                               "subsequence_begins=",
                               subseq_dim,
                               ", la_block_indices_begins=",
                               block_begins_dim,
                               ".");
    }

    // batch_size_in_sequences compatibility: processed_tokens[0] + 1 == subsequence_begins[0]
    if (subseq_static && proc_tokens_static) {
        const auto subseq_dim = subsequence_begins_ps[0];
        const auto proc_tokens_dim = processed_tokens_ps[0];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (proc_tokens_dim + 1).compatible(subseq_dim),
                               "The size of processed_tokens must be batch_size_in_sequences (subsequence_begins "
                               "size - 1). Got: processed_tokens=",
                               proc_tokens_dim,
                               ", subsequence_begins=",
                               subseq_dim,
                               ".");
    }

    // batch_size_in_sequences compatibility: cache_interval[0] + 1 == subsequence_begins[0]
    if (subseq_static && cache_int_static) {
        const auto subseq_dim = subsequence_begins_ps[0];
        const auto cache_int_dim = cache_interval_ps[0];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (cache_int_dim + 1).compatible(subseq_dim),
                               "The size of cache_interval must be batch_size_in_sequences (subsequence_begins "
                               "size - 1). Got: cache_interval=",
                               cache_int_dim,
                               ", subsequence_begins=",
                               subseq_dim,
                               ".");
    }

    // output_embeds has the same shape as input_embeds: [batch_size_in_tokens, hidden_size]
    return {input_embeds_ps};
}
}  // namespace ov::op::internal