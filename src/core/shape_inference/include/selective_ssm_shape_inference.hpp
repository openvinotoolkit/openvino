// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/selective_ssm.hpp"
#include "utils.hpp"

namespace ov::op::internal {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const SelectiveSSM* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);

    const auto& A_ps = input_shapes[0];
    const auto& dt_ps = input_shapes[1];
    const auto& B_ps = input_shapes[2];
    const auto& x_ps = input_shapes[3];
    const auto& C_ps = input_shapes[4];
    const auto& state_ps = input_shapes[5];

    const auto& batch = x_ps[0];
    const auto& seq_len = x_ps[1];
    const auto& num_heads = x_ps[2];
    const auto& head_dim = x_ps[3];
    const auto& num_groups = B_ps[2];
    const auto& state_size = B_ps[3];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dt_ps[0].compatible(batch) && B_ps[0].compatible(batch) && C_ps[0].compatible(batch) &&
                               state_ps[0].compatible(batch),
                           "The batch dimension of all inputs should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dt_ps[1].compatible(seq_len) && B_ps[1].compatible(seq_len) && C_ps[1].compatible(seq_len),
                           "The sequence length of `dt`, `B`, `x` and `C` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           A_ps[0].compatible(num_heads) && dt_ps[2].compatible(num_heads) &&
                               state_ps[1].compatible(num_heads),
                           "The number of heads of `A`, `dt`, `x` and `recurrent_state` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_ps[2].compatible(head_dim),
                           "The head dimension of `x` and `recurrent_state` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           C_ps[2].compatible(num_groups),
                           "The number of groups of `B` and `C` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           C_ps[3].compatible(state_size) && state_ps[3].compatible(state_size),
                           "The state size of `B`, `C` and `recurrent_state` should be the same.");

    if (num_heads.is_static() && num_groups.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               num_groups.get_length() != 0 && num_heads.get_length() % num_groups.get_length() == 0,
                               "The number of heads should be divisible by the number of groups.");
    }

    return {TRShape{batch, seq_len, num_heads, head_dim}, state_ps};
}

}  // namespace ov::op::internal
