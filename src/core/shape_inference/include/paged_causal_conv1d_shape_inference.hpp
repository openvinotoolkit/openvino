// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace ov::op::internal {
template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes.size() == 9);

    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[2].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[3].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[4].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[5].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[6].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[7].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[8].rank().compatible(1));

    const auto input_embeds_rank_is_static = input_shapes[0].rank().is_static();
    const auto conv_state_table_rank_is_static = input_shapes[1].rank().is_static();
    const auto conv_weight_rank_is_static = input_shapes[2].rank().is_static();
    const auto conv_bias_rank_is_static = input_shapes[3].rank().is_static();
    const auto subsequence_begins_rank_is_static = input_shapes[4].rank().is_static();
    const auto la_block_indices_rank_is_static = input_shapes[5].rank().is_static();
    const auto la_block_indices_begins_rank_is_static = input_shapes[6].rank().is_static();
    const auto processed_tokens_rank_is_static = input_shapes[7].rank().is_static();
    const auto cache_interval_rank_is_static = input_shapes[8].rank().is_static();

    if (input_embeds_rank_is_static && conv_state_table_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[0][1].compatible(input_shapes[1][1]),
                               "The hidden_size dimensions of input_embeds and conv_state_table inputs must be "
                               "compatible.");
    }

    if (conv_state_table_rank_is_static && conv_weight_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[1][2].compatible(input_shapes[2][2]),
                               "The kernel_size dimensions of conv_state_table and conv_weight inputs must be "
                               "compatible.");
    }

    if (input_embeds_rank_is_static && conv_weight_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[2][0].compatible(input_shapes[0][1]),
                               "The out_channels dimension of conv_weight must be compatible with the hidden_size "
                               "dimension of input_embeds.");
    }

    if (conv_bias_rank_is_static && conv_weight_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            input_shapes[3][0].compatible(input_shapes[2][0]) || input_shapes[3][0].compatible(ov::Dimension(0)),
            "The size of conv_bias must be compatible with the out_channels dimension of "
            "conv_weight or equal to 0 (no bias).");
    }

    if (conv_state_table_rank_is_static && la_block_indices_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[1][0].compatible(input_shapes[5][0]),
                               "The num_blocks dimension of la_block_indices must be compatible with the num_blocks "
                               "dimension of conv_state_table.");
    }

    if (subsequence_begins_rank_is_static && la_block_indices_begins_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[4][0].compatible(input_shapes[6][0]),
                               "The size of subsequence_begins must be compatible with the size of "
                               "la_block_indices_begins.");
    }

    if (subsequence_begins_rank_is_static && processed_tokens_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (input_shapes[7][0] + 1).compatible(input_shapes[4][0]),
                               "The size of processed_tokens must be batch_size_in_sequences (subsequence_begins "
                               "size - 1).");
    }

    if (subsequence_begins_rank_is_static && cache_interval_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (input_shapes[8][0] + 1).compatible(input_shapes[4][0]),
                               "The size of cache_interval must be batch_size_in_sequences (subsequence_begins "
                               "size - 1).");
    }

    // output_embeds has the same shape as input_embeds: [batch_size_in_tokens, hidden_size]
    return {input_shapes[0]};
}
}  // namespace ov::op::internal