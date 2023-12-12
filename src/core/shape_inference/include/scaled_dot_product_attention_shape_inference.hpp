// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v13 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ScaledDotProductAttention* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    const bool iscausal = op->get_causal();
    using DimType = typename T::value_type;
    const auto inputs_count = input_shapes.size();
    const auto has_attention_mask_input = inputs_count >= 4;
    const auto has_scale_input = inputs_count == 5;
    NODE_VALIDATION_CHECK(op, inputs_count == 3 || has_attention_mask_input || has_scale_input);
    TRShape n_dims{};
    DimType e_dim{};
    DimType l_dim{};
    DimType s_dim{};
    DimType ev_dim{};

    const auto query = input_shapes[0];
    if (query.rank().is_static()) {
        bool query_input_correctness = query.rank().get_length() >= 3;
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               query_input_correctness,
                               "Query input shape not compatible with other inputs.");
        n_dims = TRShape(std::vector<DimType>(query.begin(), query.end() - 2));
        l_dim = *(query.end() - 2);
        e_dim = *(query.end() - 1);
    } else {
        n_dims = query;
    }

    const auto key = input_shapes[1];
    if (key.rank().is_static()) {
        const bool key_input_correctness =
            key.rank().get_length() >= 3 &&
            TRShape::broadcast_merge_into(n_dims,
                                          TRShape(std::vector<DimType>(key.begin(), key.end() - 2)),
                                          AutoBroadcastType::NUMPY) &&
            DimType::broadcast_merge(e_dim, e_dim, *(key.end() - 1));
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               key_input_correctness,
                               "Key input shape not compatible with other inputs.");
        s_dim = *(key.end() - 2);
    }

    const auto value = input_shapes[2];
    if (value.rank().is_static()) {
        const bool value_input_correctness =
            value.rank().get_length() >= 3 &&
            TRShape::broadcast_merge_into(n_dims,
                                          TRShape(std::vector<DimType>(value.begin(), value.end() - 2)),
                                          AutoBroadcastType::NUMPY) &&
            DimType::broadcast_merge(s_dim, s_dim, *(value.end() - 2));
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               value_input_correctness,
                               "Value input shape not compatible with other inputs.");
        ev_dim = *(value.end() - 1);
    }

    if (has_attention_mask_input && !iscausal) {
        const auto attention_mask = input_shapes[3];
        if (attention_mask.rank().is_static() && attention_mask.rank() != 0) {
            const auto attention_mask_rank_len = attention_mask.rank().get_length();
            bool attention_mask_input_correctness =
                attention_mask_rank_len >= 2 && DimType::broadcast_merge(l_dim, l_dim, *(attention_mask.end() - 2)) &&
                DimType::broadcast_merge(s_dim, s_dim, *(attention_mask.end() - 1));
            if (attention_mask_rank_len >= 3) {
                attention_mask_input_correctness =
                    attention_mask_input_correctness &&
                    TRShape::broadcast_merge_into(
                        n_dims,
                        TRShape(std::vector<DimType>(attention_mask.begin(), attention_mask.end() - 2)),
                        AutoBroadcastType::NUMPY);
            }
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   attention_mask_input_correctness,
                                   "Attention mask input shape not compatible with other inputs.");
        }
    }

    if (has_scale_input) {
        if (input_shapes[4].rank().is_static() && input_shapes[4].is_static()) {
            const auto scale_is_scalar = input_shapes[4].rank().compatible(0);
            const auto scale_has_one_elem = input_shapes[4].rank().compatible(1) && input_shapes[4][0].compatible(1);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   scale_is_scalar || scale_has_one_elem,
                                   "Scale input must be scalar or have 1 element.");
        }
    }

    if (n_dims.rank().is_static()) {
        n_dims.push_back(l_dim);
        n_dims.push_back(ev_dim);
    }
    return {n_dims};
}
}  // namespace v13
}  // namespace op
}  // namespace ov
