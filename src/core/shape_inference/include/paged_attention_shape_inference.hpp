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

    // Value head_size may be not same with key
    auto out_ps = input_shapes[0];
    const auto& key_ps = input_shapes[1];
    const auto& value_ps = input_shapes[2];
    const auto& past_lens_ps = input_shapes[5];
    const auto& evictable_sizes_ps = input_shapes[22];

    // Compute for output shape
    if (out_ps.rank().is_static()) {
        if (key_ps.rank().is_static() && key_ps.rank().get_length() >= 2 && key_ps[1].is_static() &&
            value_ps.rank().is_static() && key_ps.rank().get_length() >= 2) {
            // The dim of out_ps[1] should be `num_heads * v_head_size`, it can be obtained from:
            //   q: query_ps[1] = num_heads * head_size
            //   k: key_ps[1] = num_kv_heads * head_size
            //   v: value_ps[1] = num_kv_heads * v_head_size
            // therefore:
            //   q * v / k = (num_heads * head_size) * (num_kv_heads * v_head_size) /
            //               (num_kv_heads * head_size) = num_heads * v_head_size
            out_ps[1] = out_ps[1] * value_ps[1] / key_ps[1].get_length();
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
            auto computed_dim =
                key_ps[0].get_length() + std::accumulate(past_lens.value().begin(), past_lens.value().end(), 0);
            scores_ps.push_back(computed_dim);
        } else {
            scores_ps.push_back(Dimension::dynamic());
        }
    } else {
        scores_ps.push_back(Dimension::dynamic());
    }

    auto& diversity_ps = output_shapes[2];
    // COmpute for diversity shape
    // Assumes 2D diversity [batch, group size]
    auto batch_dim = Dimension::dynamic();
    auto width_dim = Dimension::dynamic();

    // Compute the batch from the length of the evictable_sizes vector
    if (evictable_sizes_ps.rank().is_static() && evictable_sizes_ps.rank().get_length() == 1 &&
        evictable_sizes_ps[0].is_static()) {
        batch_dim = evictable_sizes_ps[0];
    } else if (key_ps.rank().is_static() && key_ps.rank().get_length() >= 1 && key_ps[0].is_static()) {
        // Fallback: batch from key tensor
        batch_dim = key_ps[0];
    }

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

    diversity_ps.push_back(batch_dim);
    diversity_ps.push_back(width_dim);

    return output_shapes;
}
}  // namespace op
}  // namespace ov
