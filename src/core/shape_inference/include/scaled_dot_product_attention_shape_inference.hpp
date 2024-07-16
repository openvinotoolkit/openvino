// Copyright (C) 2018-2024 Intel Corporation
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
    const bool& iscausal = op->get_causal();
    using DimType = typename T::value_type;
    const auto& inputs_count = input_shapes.size();
    const auto& has_attention_mask_input = inputs_count >= 4;
    const auto& has_scale_input = inputs_count == 5;
    NODE_VALIDATION_CHECK(op, inputs_count == 3 || has_attention_mask_input || has_scale_input);
    DimType e_dim{};
    DimType l_dim{};
    DimType s_dim{};
    DimType ev_dim{};

    auto output_shapes = std::vector<TRShape>{input_shapes[0]};
    auto& n_dims = output_shapes[0];
    const auto& n_dims_rank = n_dims.rank();
    if (n_dims_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               n_dims_rank.get_length() >= 3,
                               "Query input rank length must be at least 3 or more.");
        l_dim = *(n_dims.end() - 2);
        e_dim = *(n_dims.end() - 1);
        n_dims.resize(n_dims.size() - 2);
    }

    const auto& key = input_shapes[1];
    const auto& key_rank = key.rank();
    if (key_rank.is_static()) {
        const bool& key_input_correctness =
            key_rank.get_length() >= 3 &&
            TRShape::broadcast_merge_into(n_dims,
                                          TRShape(std::vector<DimType>(key.begin(), key.end() - 2)),
                                          AutoBroadcastType::NUMPY) &&
            DimType::merge(e_dim, e_dim, *(key.end() - 1));
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               key_input_correctness,
                               "Key input shape not compatible with other inputs.");
        s_dim = *(key.end() - 2);
    }

    const auto& value = input_shapes[2];
    const auto& value_rank = value.rank();
    if (value_rank.is_static()) {
        const bool& value_input_correctness =
            value_rank.get_length() >= 3 &&
            TRShape::broadcast_merge_into(n_dims,
                                          TRShape(std::vector<DimType>(value.begin(), value.end() - 2)),
                                          AutoBroadcastType::NUMPY) &&
            DimType::merge(s_dim, s_dim, *(value.end() - 2));
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               value_input_correctness,
                               "Value input shape not compatible with other inputs.");
        ev_dim = *(value.end() - 1);
    }

    if (has_attention_mask_input && !iscausal) {
        const auto& attention_mask = input_shapes[3];
        const auto& attention_mask_rank = attention_mask.rank();
        if (attention_mask_rank.is_static() && attention_mask_rank != 0) {
            const auto& attention_mask_rank_len = attention_mask_rank.get_length();
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
        const auto& scale_rank = input_shapes[4].rank();
        if (scale_rank.is_static() || input_shapes[4].is_static()) {
            const auto& scale_is_scalar = scale_rank.compatible(0);
            const auto& scale_has_one_elem = scale_rank.compatible(1) && input_shapes[4][0].compatible(1);
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
    return output_shapes;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
