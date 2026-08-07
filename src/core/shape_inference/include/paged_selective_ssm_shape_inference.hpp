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
    const auto& subsequence_begins_ps = input_shapes[6];
    const auto& la_block_indices_ps = input_shapes[7];
    const auto& la_block_indices_begins_ps = input_shapes[8];
    const auto& num_processed_tokens_ps = input_shapes[9];
    const auto& cache_interval_ps = input_shapes[10];

    const auto A_rank_is_static = A_ps.rank().is_static();
    const auto dt_rank_is_static = dt_ps.rank().is_static();
    const auto B_rank_is_static = B_ps.rank().is_static();
    const auto x_rank_is_static = x_ps.rank().is_static();
    const auto C_rank_is_static = C_ps.rank().is_static();
    const auto state_rank_is_static = state_ps.rank().is_static();
    const auto subsequence_begins_rank_is_static = subsequence_begins_ps.rank().is_static();
    const auto la_block_indices_rank_is_static = la_block_indices_ps.rank().is_static();
    const auto la_block_indices_begins_rank_is_static = la_block_indices_begins_ps.rank().is_static();
    const auto num_processed_tokens_rank_is_static = num_processed_tokens_ps.rank().is_static();
    const auto cache_interval_rank_is_static = cache_interval_ps.rank().is_static();

    Dimension token_dim, num_heads_dim, head_dim_dim, state_size_dim;

    bool token_ok = true;
    if (x_rank_is_static)
        token_ok &= Dimension::merge(token_dim, token_dim, x_ps[0]);
    if (dt_rank_is_static)
        token_ok &= Dimension::merge(token_dim, token_dim, dt_ps[0]);
    if (B_rank_is_static)
        token_ok &= Dimension::merge(token_dim, token_dim, B_ps[0]);
    if (C_rank_is_static)
        token_ok &= Dimension::merge(token_dim, token_dim, C_ps[0]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           token_ok,
                           "The token dimension of `dt`, `B`, `x` and `C` should be the same.");

    bool num_heads_ok = true;
    if (x_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, x_ps[1]);
    if (A_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, A_ps[0]);
    if (dt_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, dt_ps[1]);
    if (state_rank_is_static)
        num_heads_ok &= Dimension::merge(num_heads_dim, num_heads_dim, state_ps[1]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           num_heads_ok,
                           "The number of heads of `A`, `dt`, `x` and `recurrent_state_table` should be the same.");

    if (B_rank_is_static && C_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               C_ps[1].compatible(B_ps[1]),
                               "The number of groups of `B` and `C` should be the same.");
    }

    bool head_dim_ok = true;
    if (x_rank_is_static)
        head_dim_ok &= Dimension::merge(head_dim_dim, head_dim_dim, x_ps[2]);
    if (state_rank_is_static)
        head_dim_ok &= Dimension::merge(head_dim_dim, head_dim_dim, state_ps[2]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           head_dim_ok,
                           "The head dimension of `x` and `recurrent_state_table` should be the same.");

    bool state_size_ok = true;
    if (state_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, state_ps[3]);
    if (B_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, B_ps[2]);
    if (C_rank_is_static)
        state_size_ok &= Dimension::merge(state_size_dim, state_size_dim, C_ps[2]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_size_ok,
                           "The state size of `B`, `C` and `recurrent_state_table` should be the same.");

    if (num_heads_dim.is_static() && B_rank_is_static) {
        const auto& num_groups = B_ps[1];
        if (num_groups.is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                num_groups.get_length() != 0 && num_heads_dim.get_length() % num_groups.get_length() == 0,
                "The number of heads should be divisible by the number of groups.");
        }
    }

    if (la_block_indices_rank_is_static && state_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               la_block_indices_ps[0].compatible(state_ps[0]),
                               "The number of blocks of `la_block_indices` and `recurrent_state_table` should "
                               "be the same.");
    }
    if (subsequence_begins_rank_is_static && la_block_indices_begins_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               subsequence_begins_ps[0].compatible(la_block_indices_begins_ps[0]),
                               "The number of sequences of `subsequence_begins` and `la_block_indices_begins` "
                               "should be the same.");
    }
    if (num_processed_tokens_rank_is_static && cache_interval_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               num_processed_tokens_ps[0].compatible(cache_interval_ps[0]),
                               "The number of sequences of `num_processed_tokens` and `cache_interval` should "
                               "be the same.");
    }
    if (subsequence_begins_rank_is_static && num_processed_tokens_rank_is_static) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               (subsequence_begins_ps[0] - Dimension(1)).compatible(num_processed_tokens_ps[0]),
                               "The number of sequences of `subsequence_begins` and `num_processed_tokens` "
                               "should be the same.");
    }

    auto output_shapes = std::vector<TRShape>{x_ps};
    if (x_rank_is_static) {
        output_shapes[0] = TRShape{token_dim, num_heads_dim, head_dim_dim};
    }
    return output_shapes;
}

}  // namespace ov::op::internal
