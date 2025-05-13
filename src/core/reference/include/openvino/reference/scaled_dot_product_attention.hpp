// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cmath>

#include "openvino/core/shape.hpp"
#include "openvino/reference/add.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/softmax.hpp"
#include "openvino/reference/transpose.hpp"

// def ScaledDotProductAttention(query, key, value, attn_mask=None, scale=None,
// *, causal):
//     L, S = Gather(ShapeOf(query), -2), Gather(ShapeOf(key), -2)
//     if scale is None:
//         scale = 1.0 / Sqrt(ConvertLike(Gather(ShapeOf(query), -1), query))
//     attn_bias = Broadcast(ConvertLike(0, query), [L, S])
//     if causal:
//         attn_bias = numpy.triu(Broadcast(ConvertLike(-inf, query), [L, S]),
//         k=1)
//     elif attn_mask is not None:
//         if attn_mask.element_type == boolean:
//             attn_bias = Select(LogicalNot(attn_mask), ConvertLike(-inf,
//             query), ConvertLike(0, query))
//         else:
//             attn_bias += attn_mask
//     attn_weight = MatMul(query, Transpose(key, [-2, -1])) * scale
//     attn_weight += attn_bias
//     attn_weight = Softmax(attn_weight, axis=-1)
//     return MatMul(attn_weight, value)

namespace helpers {
template <typename T>
std::vector<T> GetAttentionMask(const ov::Shape& shape) {
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
}  // namespace helpers
namespace ov {
namespace reference {
template <typename T>
void scaled_dot_product_attention(const T* query,
                                  const T* key,
                                  const T* value,
                                  const T* mask,
                                  const T* scale,
                                  T* output,
                                  bool is_causal,
                                  const Shape& query_shape,
                                  const Shape& key_shape,
                                  const Shape& value_shape,
                                  const Shape& mask_shape,
                                  const Shape& output_shape) {
    const T* bias = nullptr;
    Shape biasShape = {};

    if (mask) {
        bias = mask;
        biasShape = mask_shape;
    }

    std::vector<T> attentionMaskData;
    if (is_causal) {
        auto L = query_shape[query_shape.size() - 2];
        auto S = key_shape[key_shape.size() - 2];
        biasShape = {L, S};
        attentionMaskData = helpers::GetAttentionMask<T>(biasShape);
        bias = attentionMaskData.data();
    }

    const float defaultScaleVal = 1.0f / static_cast<float>(std::sqrt(query_shape[query_shape.size() - 1]));
    const T scaleVal = scale ? *scale : static_cast<T>(defaultScaleVal);

    auto keyTransposedShape = key_shape;
    std::swap(keyTransposedShape[key_shape.size() - 2], keyTransposedShape[key_shape.size() - 1]);

    std::vector<T> keyTransposed(key, key + shape_size(key_shape));
    std::vector<int64_t> axesOrder;
    for (size_t i = 0; i < key_shape.size(); ++i) {
        axesOrder.push_back(i);
    }
    std::swap(axesOrder[key_shape.size() - 2], axesOrder[key_shape.size() - 1]);

    ov::reference::transpose(reinterpret_cast<const char*>(key),
                             reinterpret_cast<char*>(keyTransposed.data()),
                             key_shape,
                             sizeof(T),
                             axesOrder,
                             keyTransposedShape);

    auto qkShape = query_shape;
    qkShape[qkShape.size() - 1] = key_shape[key_shape.size() - 2];

    std::vector<T> qkData(shape_size(qkShape), 0);
    ov::reference::matmul<T>(query,
                             keyTransposed.data(),
                             qkData.data(),
                             query_shape,
                             keyTransposedShape,
                             qkShape,
                             false,
                             false);

    std::transform(qkData.begin(), qkData.end(), qkData.begin(), [scaleVal](T& val) {
        return val * scaleVal;
    });

    if (bias) {
        ov::reference::add<T>(qkData.data(), bias, qkData.data(), qkShape, biasShape, ov::op::AutoBroadcastType::NUMPY);
    }

    std::vector<T> qkDataSoftmax(qkData.size(), 0);
    ov::reference::softmax<T>(qkData.data(), qkDataSoftmax.data(), qkShape, ov::AxisSet{qkShape.size() - 1});
    ov::reference::matmul<T>(qkDataSoftmax.data(), value, output, qkShape, value_shape, output_shape, false, false);
}

}  // namespace reference
}  // namespace ov