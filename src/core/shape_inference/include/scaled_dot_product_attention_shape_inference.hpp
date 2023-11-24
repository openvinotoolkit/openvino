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
std::vector<TRShape> shape_infer(const ScaledDotProductAttention* op, const std::vector<T>& input_shapes) {
    const auto inputs_count = input_shapes.size();
    const auto has_attention_mask_input = inputs_count == 4;
    const auto has_scale_input = inputs_count == 5;
    NODE_VALIDATION_CHECK(op, inputs_count == 3 || has_attention_mask_input || has_scale_input);
    TRShape n_dims{TRShape::dynamic()};
    Dimension s_dim{Dimension::dynamic()};
    Dimension e_dim{Dimension::dynamic()};
    Dimension l_dim{Dimension::dynamic()};
    Dimension ev_dim{Dimension::dynamic()};
    TRShape m_dims{TRShape::dynamic()};

    const auto query = input_shapes[0];
    if (query.rank().is_static()) {
        OPENVINO_ASSERT(query.rank().get_length() >= 3);
        OPENVINO_ASSERT(
            TRShape::merge_into(n_dims, TRShape(std::vector<Dimension>(query.begin(), query.end() - 2))));
        OPENVINO_ASSERT(Dimension::merge(l_dim, l_dim, *(query.end() - 2)));
        OPENVINO_ASSERT(Dimension::merge(e_dim, e_dim, *(query.end() - 1)));
    }

    const auto key = input_shapes[1];
    if (key.rank().is_static()) {
        OPENVINO_ASSERT(key.rank().get_length() >= 3);
        OPENVINO_ASSERT(
            TRShape::merge_into(n_dims, TRShape(std::vector<Dimension>(key.begin(), key.end() - 2))));
        OPENVINO_ASSERT(Dimension::merge(s_dim, s_dim, *(key.end() - 2)));
        OPENVINO_ASSERT(Dimension::merge(e_dim, e_dim, *(key.end() - 1)));
    }

    const auto value = input_shapes[2];
    if (value.rank().is_static()) {
        OPENVINO_ASSERT(value.rank().get_length() >= 3);
        OPENVINO_ASSERT(
            TRShape::merge_into(n_dims, TRShape(std::vector<Dimension>(value.begin(), value.end() - 2))));
        OPENVINO_ASSERT(Dimension::merge(s_dim, s_dim, *(value.end() - 2)));
        OPENVINO_ASSERT(Dimension::merge(ev_dim, ev_dim, *(value.end() - 1)));
    }

    if (has_attention_mask_input && !has_scale_input) {
        const auto attention_mask = input_shapes[3];
        if (attention_mask.rank().is_static() && attention_mask.rank() != 0) {
            OPENVINO_ASSERT(attention_mask.rank().get_length() >= 3);
            OPENVINO_ASSERT(TRShape::merge_into(
                                m_dims,
                                TRShape(std::vector<Dimension>(attention_mask.begin(), attention_mask.end() - 2))));
            OPENVINO_ASSERT(Dimension::merge(l_dim, l_dim, *(attention_mask.end() - 2)));
            OPENVINO_ASSERT(Dimension::merge(s_dim, s_dim, *(attention_mask.end() - 1)));
            OPENVINO_ASSERT(TRShape::broadcast_merge_into(m_dims, n_dims, AutoBroadcastType::NUMPY));
            OPENVINO_ASSERT(n_dims.compatible(m_dims));
        }
    }

    if (has_scale_input) {
        const auto scale = input_shapes[4];
        OPENVINO_ASSERT(scale.rank().is_dynamic() || scale.rank() == 0, "Scale input accepts only scalar tensor.");
    }

    TRShape out_pshape = n_dims;
    out_pshape.push_back(l_dim);
    out_pshape.push_back(ev_dim);
    return {out_pshape};
}
}  // namespace v13
}  // namespace op
}  // namespace ov
