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
                           input_shapes.size() == 11,
                           "Incorrect number of inputs for PagedGatedDeltaNet");

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

    const auto q_head_num = query_ps[1];
    const auto k_head_num = key_ps[1];
    const auto v_head_num = value_ps[1];

    const auto q_head_size = query_ps[2];
    const auto k_head_size = key_ps[2];
    const auto v_head_size = value_ps[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           q_head_num.compatible(k_head_num),
                           "The number of heads in query and key should be the same, but got ",
                           q_head_num,
                           " and ",
                           k_head_num,
                           ".");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           k_head_size.compatible(q_head_size),
                           "The head size in key and query should be the same, but got ",
                           k_head_size,
                           " and ",
                           q_head_size,
                           ".");

    const auto gate_head_num = gate_ps[1];
    const auto beta_head_num = beta_ps[1];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           gate_head_num.compatible(v_head_num) && beta_head_num.compatible(v_head_num),
                           "The number of heads in gate, beta, and value should be the same, but got gate=",
                           gate_head_num,
                           ", beta=",
                           beta_head_num,
                           ", value=",
                           v_head_num,
                           ".");

    // [num_blocks, v_num_heads, value_head_dim, key_head_dim]
    const auto state_head_num = state_ps[1];
    const auto state_key_dim = state_ps[3];
    const auto state_value_dim = state_ps[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_head_num.compatible(v_head_num),
                           "The number of heads in recurrent_state_table and value should be the same, but got ",
                           state_head_num,
                           " and ",
                           v_head_num,
                           ".");

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        state_key_dim.compatible(k_head_size),
        "The key_head_dim of recurrent_state_table and head size of key should be the same, but got ",
        state_key_dim,
        " and ",
        k_head_size,
        ".");

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        state_value_dim.compatible(v_head_size),
        "The value_head_dim of recurrent_state_table and head size of value should be the same, but got ",
        state_value_dim,
        " and ",
        v_head_size,
        ".");

    // output: [batch_size_in_tokens, v_num_heads, value_head_dim] — same shape as value input
    return {value_ps};
}
}  // namespace ov::op::internal
