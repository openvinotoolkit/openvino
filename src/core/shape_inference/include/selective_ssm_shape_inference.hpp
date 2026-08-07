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

    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[2].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[3].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[4].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[5].rank().compatible(4));

    const auto& A_ps = input_shapes[0];
    const auto& dt_ps = input_shapes[1];
    const auto& B_ps = input_shapes[2];
    const auto& x_ps = input_shapes[3];
    const auto& C_ps = input_shapes[4];
    const auto& state_ps = input_shapes[5];

    const auto A_rank_is_static = A_ps.rank().is_static();
    const auto dt_rank_is_static = dt_ps.rank().is_static();
    const auto B_rank_is_static = B_ps.rank().is_static();
    const auto x_rank_is_static = x_ps.rank().is_static();
    const auto C_rank_is_static = C_ps.rank().is_static();
    const auto state_rank_is_static = state_ps.rank().is_static();

    Dimension batch_dim, seq_len_dim, num_heads_dim, head_dim_dim, state_size_dim;

    bool batch_ok = true;
    if (x_rank_is_static)
        batch_ok &= Dimension::merge(batch_dim, batch_dim, x_ps[0]);
    if (dt_rank_is_static)
        batch_ok &= Dimension::merge(batch_dim, batch_dim, dt_ps[0]);
    if (B_rank_is_static)
        batch_ok &= Dimension::merge(batch_dim, batch_dim, B_ps[0]);
    if (C_rank_is_static)
        batch_ok &= Dimension::merge(batch_dim, batch_dim, C_ps[0]);
    if (state_rank_is_static)
        batch_ok &= Dimension::merge(batch_dim, batch_dim, state_ps[0]);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, batch_ok, "The batch dimension of all inputs should be the same.");

    bool seq_len_ok = true;
    if (x_rank_is_static)
        seq_len_ok &= Dimension::merge(seq_len_dim, seq_len_dim, x_ps[1]);
    if (dt_rank_is_static)
        seq_len_ok &= Dimension::merge(seq_len_dim, seq_len_dim, dt_ps[1]);
    if (B_rank_is_static)
        seq_len_ok &= Dimension::merge(seq_len_dim, seq_len_dim, B_ps[1]);
    if (C_rank_is_static)
        seq_len_ok &= Dimension::merge(seq_len_dim, seq_len_dim, C_ps[1]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           seq_len_ok,
                           "The sequence length of `dt`, `B`, `x` and `C` should be the same.");

    bool num_heads_ok = true;
    if (x_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, x_ps[2]);
    if (A_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, A_ps[0]);
    if (dt_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, dt_ps[2]);
    if (state_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, state_ps[1]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           num_heads_ok,
                           "The number of heads of `A`, `dt`, `x` and `recurrent_state` should be the same.");

    bool head_dim_ok = true;
    if (x_rank_is_static)
        head_dim_ok &= Dimension::merge(head_dim_dim, head_dim_dim, x_ps[3]);
    if (state_rank_is_static)
        head_dim_ok &= Dimension::merge(head_dim_dim, head_dim_dim, state_ps[2]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           head_dim_ok,
                           "The head dimension of `x` and `recurrent_state` should be the same.");

    if (B_rank_is_static && C_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               C_ps[2].compatible(B_ps[2]),
                               "The number of groups of `B` and `C` should be the same.");
    }

    bool state_size_ok = true;
    if (state_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, state_ps[3]);
    if (B_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, B_ps[3]);
    if (C_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, C_ps[3]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_size_ok,
                           "The state size of `B`, `C` and `recurrent_state` should be the same.");

    if (num_heads_dim.is_static() && B_rank_is_static) {
        const auto& num_groups = B_ps[2];
        if (num_groups.is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                num_groups.get_length() != 0 && num_heads_dim.get_length() % num_groups.get_length() == 0,
                "The number of heads should be divisible by the number of groups.");
        }
    }

    auto output_shapes = std::vector<TRShape>{x_ps, state_ps};
    if (x_rank_is_static) {
        output_shapes[0] = TRShape{batch_dim, seq_len_dim, num_heads_dim, head_dim_dim};
    }
    if (state_rank_is_static) {
        output_shapes[1] = TRShape{batch_dim, num_heads_dim, head_dim_dim, state_size_dim};
    }
    return output_shapes;
}

}  // namespace ov::op::internal
