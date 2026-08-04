// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/paged_selective_ssm.hpp"
#include "utils.hpp"

namespace ov::op::internal {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const PagedSelectiveSSM* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 11);

    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(2));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[2].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[3].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[4].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[5].rank().compatible(4));
    for (size_t i = 6; i < 11; ++i) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[i].rank().compatible(1));
    }

    const auto& A_ps = input_shapes[0];
    const auto& dt_ps = input_shapes[1];
    const auto& B_ps = input_shapes[2];
    const auto& x_ps = input_shapes[3];
    const auto& C_ps = input_shapes[4];
    const auto& state_ps = input_shapes[5];

    const auto& batch_tokens = x_ps[0];
    const auto& num_heads = x_ps[1];
    const auto& head_dim = x_ps[2];
    const auto& num_groups = B_ps[1];
    const auto& state_size = B_ps[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dt_ps[0].compatible(batch_tokens) && B_ps[0].compatible(batch_tokens) &&
                               C_ps[0].compatible(batch_tokens),
                           "The token dimension of `dt`, `B`, `x` and `C` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           A_ps[0].compatible(num_heads) && dt_ps[1].compatible(num_heads) &&
                               state_ps[1].compatible(num_heads),
                           "The number of heads of `A`, `dt`, `x` and `recurrent_state_table` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           C_ps[1].compatible(num_groups),
                           "The number of groups of `B` and `C` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_ps[2].compatible(head_dim),
                           "The head dimension of `x` and `recurrent_state_table` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           C_ps[2].compatible(state_size) && state_ps[3].compatible(state_size),
                           "The state size of `B`, `C` and `recurrent_state_table` should be the same.");

    if (num_heads.is_static() && num_groups.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               num_groups.get_length() != 0 && num_heads.get_length() % num_groups.get_length() == 0,
                               "The number of heads should be divisible by the number of groups.");
    }

    return {x_ps};
}

}  // namespace ov::op::internal
