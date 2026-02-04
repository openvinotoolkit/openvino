// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/paged_attention.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const PagedAttentionExtension* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 25, "Expected exactly 25 inputs but got ", input_shapes.size());
    auto output_shapes = std::vector<TRShape>(3);

    // Output[0] feature dim is `num_heads * v_head_size`.
    auto out_ps = input_shapes[0];
    const auto& key_ps = input_shapes[1];
    const auto& value_ps = input_shapes[2];
    const auto& past_lens_ps = input_shapes[5];
    const auto& evictable_sizes_ps = input_shapes[22];

    // Compute for output shape
    if (out_ps.rank().is_static() && out_ps.rank().get_length() >= 2 && out_ps[1].is_static()) {
        if (key_ps.rank().is_static() && key_ps.rank().get_length() >= 2 && key_ps[1].is_static() &&
            value_ps.rank().is_static() && value_ps.rank().get_length() >= 2 && value_ps[1].is_static()) {
            // The dim of out_ps[1] should be `num_heads * v_head_size`, it can be obtained from:
            //   q: query_ps[1] = num_heads * head_size
            //   k: key_ps[1] = num_kv_heads * head_size
            //   v: value_ps[1] = num_kv_heads * v_head_size
            // therefore:
            //   q * v / k = (num_heads * head_size) * (num_kv_heads * v_head_size) /
            //               (num_kv_heads * head_size) = num_heads * v_head_size
            // q_features * v_features / k_features = (Hq*Dk) * (Hkv*Dv) / (Hkv*Dk) = Hq*Dv
            const auto q = out_ps[1].get_length();
            const auto k = key_ps[1].get_length();
            const auto v = value_ps[1].get_length();
            NODE_VALIDATION_CHECK(op, q * v % k == 0, "Key dims cannot be zero");
            out_ps[1] = q * v / k;
            NODE_VALIDATION_CHECK(op,
                                  !ov::util::dim::is_empty(out_ps[1]),
                                  "The last dimension of output should not be empty.");
        } else {
            out_ps[1] = Dimension::dynamic();
        }
    }
    output_shapes[0] = out_ps;

    auto& scores_ps = output_shapes[1];
    // Compute for scores shape
    if (past_lens_ps.rank().is_static() && key_ps.rank().is_static() && key_ps.rank().get_length() >= 1 &&
        key_ps[0].is_static()) {
        const auto& past_lens = get_input_const_data_as<TRShape, int32_t>(op, 5, ta);
        if (past_lens.has_value()) {
            const auto token_count = key_ps[0].get_length();
            const auto past_sum = std::accumulate(past_lens.value().begin(), past_lens.value().end(), 0);
            const auto computed_dim = token_count + past_sum;
            scores_ps.push_back(computed_dim);
        } else {
            scores_ps.push_back(Dimension::dynamic());
        }
    } else {
        scores_ps.push_back(Dimension::dynamic());
    }

    auto& diversity_ps = output_shapes[2];
    // COmpute for diversity shape
    // Assumes 1D diversity [max_group_size]
    auto width_dim = Dimension::dynamic();

    // Backup in case diversity is 2D [batch_sequences, max group size]
    //
    // Batch dimension for output[2] corresponds to the number of sequences.
    // auto batch_dim = Dimension::dynamic();
    // if (past_lens_ps.rank().is_static() && past_lens_ps.rank().get_length() == 1 && past_lens_ps[0].is_static()) {
    //     batch_dim = past_lens_ps[0];
    // } else if (evictable_sizes_ps.rank().is_static() && evictable_sizes_ps.rank().get_length() == 1 &&
    //            evictable_sizes_ps[0].is_static()) {
    //     // Fallback: infer from evictable_sizes length.
    //     batch_dim = evictable_sizes_ps[0];
    // }
    // diversity_ps.push_back(batch_dim);

    // If evictable_sizes is constant, compute max for the padded width
    if (evictable_sizes_ps.rank().is_static() && evictable_sizes_ps.rank().get_length() == 1) {
        const auto& evictable_sizes = get_input_const_data_as<TRShape, int32_t>(op, 22, ta);
        if (evictable_sizes.has_value() && !evictable_sizes.value().empty()) {
            int32_t max_v = 0;
            for (const auto v : evictable_sizes.value()) {
                max_v = std::max(max_v, v);
            }
            width_dim = static_cast<int64_t>(max_v);
        }
    }

    diversity_ps.push_back(width_dim);

    return output_shapes;
}
}  // namespace op
}  // namespace ov
