// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cmath>

#include "openvino/core/shape.hpp"
#include "openvino/reference/add.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/softmax.hpp"
namespace ov::reference {
namespace helpers {
template <typename T>
std::vector<T> create_causal_attention_mask(const ov::Shape& shape) {
    std::vector<T> mask_data(shape_size(shape), std::numeric_limits<T>::lowest());
    const auto L = shape[shape.size() - 2];
    const auto S = shape[shape.size() - 1];
    for (size_t i = 0; i < L; ++i) {
        size_t j = 0;
        while (i >= j && j < S) {
            mask_data[i * S + j] = 0;
            ++j;
        }
    }
    return mask_data;
}

template <typename T>
std::vector<T> create_attention_mask_from_ov_boolean(const char* mask_bool, const ov::Shape& shape) {
    std::vector<T> mask_data(shape_size(shape));

    std::transform(mask_bool, mask_bool + shape_size(shape), mask_data.begin(), [](char val) {
        if (val == 0) {
            return std::numeric_limits<T>::lowest();
        } else {
            return static_cast<T>(0);
        }
    });

    return mask_data;
}
}  // namespace helpers
}  // namespace ov::reference

namespace ov {
namespace reference {
template <typename T, typename TMask>
void scaled_dot_product_attention(const T* query,
                                  const T* key,
                                  const T* value,
                                  const TMask* mask,
                                  const T* scale,
                                  T* output,
                                  bool is_causal,
                                  const Shape& query_shape,
                                  const Shape& key_shape,
                                  const Shape& value_shape,
                                  const Shape& mask_shape,
                                  const Shape& output_shape) {
    static_assert(std::is_same_v<T, TMask> || std::is_same_v<TMask, char>,
                  "T and TMask must be either the same type, or the TMask must be char(ov::element::boolean)");

    const T* bias = nullptr;
    Shape bias_shape = {};

    std::vector<T> attention_mask_data;
    if (mask && !is_causal) {
        bias_shape = mask_shape;
        if constexpr (std::is_same<TMask, char>::value) {
            attention_mask_data = helpers::create_attention_mask_from_ov_boolean<T>(mask, bias_shape);
            bias = attention_mask_data.data();
        } else {
            bias = mask;
        }
    }

    if (is_causal) {
        const auto L = query_shape[query_shape.size() - 2];
        const auto S = key_shape[key_shape.size() - 2];
        bias_shape = {L, S};
        attention_mask_data = helpers::create_causal_attention_mask<T>(bias_shape);
        bias = attention_mask_data.data();
    }

    const float default_scale_val = 1.0f / static_cast<float>(std::sqrt(query_shape[query_shape.size() - 1]));
    const T scale_val = scale ? *scale : static_cast<T>(default_scale_val);

    auto qk_shape = query_shape;
    qk_shape[qk_shape.size() - 1] = key_shape[key_shape.size() - 2];

    std::vector<T> qk_data(shape_size(qk_shape), 0);
    ov::reference::matmul<T>(query, key, qk_data.data(), query_shape, key_shape, qk_shape, false, true);

    ov::reference::multiply<T>(qk_data.data(),
                               &scale_val,
                               qk_data.data(),
                               qk_shape,
                               Shape{1},
                               ov::op::AutoBroadcastType::NUMPY);

    if (bias) {
        ov::reference::add<T>(qk_data.data(),
                              bias,
                              qk_data.data(),
                              qk_shape,
                              bias_shape,
                              ov::op::AutoBroadcastType::NUMPY);
    }

    std::vector<T> qk_data_softmax(qk_data.size(), 0);
    ov::reference::softmax<T>(qk_data.data(), qk_data_softmax.data(), qk_shape, ov::AxisSet{qk_shape.size() - 1});
    ov::reference::matmul<T>(qk_data_softmax.data(), value, output, qk_shape, value_shape, output_shape, false, false);
}

}  // namespace reference
}  // namespace ov