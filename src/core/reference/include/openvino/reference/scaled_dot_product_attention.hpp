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

namespace helpers {
template <typename T>
std::vector<T> CreateCausalAttentionMask(const ov::Shape& shape) {
    std::vector<T> maskData(shape_size(shape), std::numeric_limits<T>::lowest());
    auto L = shape[shape.size() - 2];
    auto S = shape[shape.size() - 1];
    for (size_t i = 0; i < L; ++i) {
        size_t j = 0;
        while (i >= j && j < S) {
            maskData[i * S + j] = 0;
            ++j;
        }
    }
    return maskData;
}

template <typename T>
std::vector<T> CreateAttentionMaskFromOvBoolean(const char* maskBool, const ov::Shape& shape) {
    std::vector<T> maskData(shape_size(shape));

    std::transform(maskBool, maskBool + shape_size(shape), maskData.begin(), [](char val) {
        if (val == 0) {
            return std::numeric_limits<T>::lowest();
        } else {
            return static_cast<T>(0);
        }
    });

    return maskData;
}
}  // namespace helpers
namespace ov {
namespace reference {
template <typename T, typename TMask>
void scaled_dot_product_attention(const T* query,
                                  const T* key,
                                  const T* value,
                                  const TMask* mask,
                                  const T* scale,
                                  T* output,
                                  bool isCausal,
                                  const Shape& queryShape,
                                  const Shape& keyShape,
                                  const Shape& valueShape,
                                  const Shape& maskShape,
                                  const Shape& outputShape) {
    static_assert(std::is_same<T, TMask>::value || std::is_same<TMask, char>::value,
                  "T and TMask must be either the same type, or the TMask must be char(ov::element::boolean)");

    const T* bias = nullptr;
    Shape biasShape = {};

    std::vector<T> attentionMaskData;
    if (mask && !isCausal) {
        biasShape = maskShape;
        if constexpr (std::is_same<TMask, char>::value) {
            attentionMaskData = helpers::CreateAttentionMaskFromOvBoolean<T>(mask, biasShape);
            bias = attentionMaskData.data();
        } else {
            bias = mask;
        }
    }

    if (isCausal) {
        const auto L = queryShape[queryShape.size() - 2];
        const auto S = keyShape[keyShape.size() - 2];
        biasShape = {L, S};
        attentionMaskData = helpers::CreateCausalAttentionMask<T>(biasShape);
        bias = attentionMaskData.data();
    }

    const float defaultScaleVal = 1.0f / static_cast<float>(std::sqrt(queryShape[queryShape.size() - 1]));
    const T scaleVal = scale ? *scale : static_cast<T>(defaultScaleVal);

    auto qkShape = queryShape;
    qkShape[qkShape.size() - 1] = keyShape[keyShape.size() - 2];

    std::vector<T> qkData(shape_size(qkShape), 0);
    ov::reference::matmul<T>(query, key, qkData.data(), queryShape, keyShape, qkShape, false, true);

    ov::reference::multiply<T>(qkData.data(),
                               &scaleVal,
                               qkData.data(),
                               qkShape,
                               Shape{1},
                               ov::op::AutoBroadcastType::NUMPY);

    if (bias) {
        ov::reference::add<T>(qkData.data(), bias, qkData.data(), qkShape, biasShape, ov::op::AutoBroadcastType::NUMPY);
    }

    std::vector<T> qkDataSoftmax(qkData.size(), 0);
    ov::reference::softmax<T>(qkData.data(), qkDataSoftmax.data(), qkShape, ov::AxisSet{qkShape.size() - 1});
    ov::reference::matmul<T>(qkDataSoftmax.data(), value, output, qkShape, valueShape, outputShape, false, false);
}

}  // namespace reference
}  // namespace ov