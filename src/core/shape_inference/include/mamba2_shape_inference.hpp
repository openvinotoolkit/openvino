// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/mamba2.hpp"
#include "utils.hpp"

namespace ov::op::internal {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Mamba2* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& dA_ps = input_shapes[0];     // [batch, num_heads, seq_len, 1, 1]
    const auto& dBx_ps = input_shapes[1];    // [batch, num_heads, seq_len, head_dim, state_size]
    const auto& C_ps = input_shapes[2];      // [batch, num_heads, seq_len, state_size]
    const auto& state_ps = input_shapes[3];  // [batch, num_heads, head_dim, state_size]

    const auto& batch = dBx_ps[0];
    const auto& num_heads = dBx_ps[1];
    const auto& seq_len = dBx_ps[2];
    const auto& head_dim = dBx_ps[3];
    const auto& state_size = dBx_ps[4];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dA_ps[0].compatible(batch) && C_ps[0].compatible(batch) && state_ps[0].compatible(batch),
                           "The batch dimension of all inputs should be the same.");

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        dA_ps[1].compatible(num_heads) && C_ps[1].compatible(num_heads) && state_ps[1].compatible(num_heads),
        "The number of heads of all inputs should be the same.");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dA_ps[2].compatible(seq_len) && C_ps[2].compatible(seq_len),
                           "The sequence length of `dA`, `dBx` and `C` should be the same.");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_ps[2].compatible(head_dim),
                           "The head dimension of `dBx` and `recurrent_state` should be the same.");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           C_ps[3].compatible(state_size) && state_ps[3].compatible(state_size),
                           "The state size of `dBx`, `C` and `recurrent_state` should be the same.");

    // output: [batch, num_heads, seq_len, head_dim]
    TRShape output_shape{batch, num_heads, seq_len, head_dim};
    // output_recurrent_state has the same shape as the input recurrent_state
    return {std::move(output_shape), state_ps};
}
}  // namespace ov::op::internal
