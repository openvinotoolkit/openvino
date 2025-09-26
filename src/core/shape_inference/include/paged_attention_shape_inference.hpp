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
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 17 || input_shapes.size() == 20);
    auto output_shapes = std::vector<TRShape>(2);

    // Value head_size may be not same with key
    auto out_ps = input_shapes[0];
    const auto& key_ps = input_shapes[1];
    const auto& value_ps = input_shapes[2];
    const auto& past_lens_ps = input_shapes[5];

    // Compute for output shape
    if (out_ps.rank().is_static()) {
        if (key_ps.rank().is_static() && value_ps.rank().is_static() && key_ps[1].is_static()) {
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
    if (past_lens_ps.rank().is_static() && key_ps.rank().is_static()) {
        const auto& past_lens = get_input_const_data_as<TRShape, int32_t>(op, 5, ta);
        NODE_VALIDATION_CHECK(op, past_lens.has_value(), "Failed to obtain past_lens values as a vector.");
        auto computed_dim =
            key_ps[0].get_length() + std::accumulate(past_lens.value().begin(), past_lens.value().end(), 0);
        scores_ps.push_back(computed_dim);
    } else {
        scores_ps.push_back(Dimension::dynamic());
    }

    return output_shapes;
}
}  // namespace op
}  // namespace ov
