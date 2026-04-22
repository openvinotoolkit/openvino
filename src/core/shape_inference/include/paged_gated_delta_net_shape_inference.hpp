// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace ov::op::internal {

template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[2].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[3].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[4].rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[5].rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[6].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[7].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[8].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[9].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[10].rank().compatible(1));

    // [batch_size_in_tokens, num_heads, key_head_dim]
    const auto& query_ps = input_shapes[0];
    const auto& key_ps = input_shapes[1];
    // [batch_size_in_tokens, v_num_heads, value_head_dim]
    const auto& value_ps = input_shapes[2];
    // [num_blocks, v_num_heads, key_head_dim, value_head_dim]
    const auto& state_ps = input_shapes[3];
    // [batch_size_in_tokens, v_num_heads]
    const auto& gate_ps = input_shapes[4];
    const auto& beta_ps = input_shapes[5];

    const auto query_rank_is_static = query_ps.rank().is_static();
    const auto key_rank_is_static = key_ps.rank().is_static();
    const auto value_rank_is_static = value_ps.rank().is_static();
    const auto state_rank_is_static = state_ps.rank().is_static();
    const auto gate_rank_is_static = gate_ps.rank().is_static();
    const auto beta_rank_is_static = beta_ps.rank().is_static();

    if (query_rank_is_static && key_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               query_ps[1].compatible(key_ps[1]),
                               "The number of heads in query and key inputs must be equal.");

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               key_ps[2].compatible(query_ps[2]),
                               "The head size of query and key inputs must be equal.");
    }

    if (value_rank_is_static && gate_rank_is_static && beta_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               gate_ps[1].compatible(value_ps[1]) && beta_ps[1].compatible(value_ps[1]),
                               "The number of heads in gate, beta, and value inputs must be equal.");
    }

    if (state_rank_is_static && value_rank_is_static && key_rank_is_static) {
        // [num_blocks, v_num_heads, key_head_dim, value_head_dim]
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               state_ps[1].compatible(value_ps[1]),
                               "The number of heads in recurrent_state_table and value inputs must be equal.");

        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            state_ps[2].compatible(key_ps[2]),
            "The key dimension of recurrent_state_table and the head size of key input must be equal.");

        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            state_ps[3].compatible(value_ps[2]),
            "The value dimension of recurrent_state_table and the head size of value input must be equal.");
    }

    // output: [batch_size_in_tokens, v_num_heads, value_head_dim] — same shape as value input
    return {value_ps};
}
}  // namespace ov::op::internal
