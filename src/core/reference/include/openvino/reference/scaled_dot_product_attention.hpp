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

namespace helpers {
template <typename T>
std::vector<T> CreateAttentionMask(const ov::Shape& shape) {
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

// --------------------------------------------------------
// This overloads are intentionally not implemented! If the linker ever complains about them,
// it means that something went wrong in scaled_dot_product_attention().
template <typename T>
const T* GetRawPtr(const char* maskBool);

template <typename T>
std::vector<T> CreateAttentionMaskFromBool(const T* maskBool, const ov::Shape& shape);
// --------------------------------------------------------

template <typename T>
typename std::enable_if<!std::is_same<T, char>::value, const T*>::type GetRawPtr(const T* maskBool) {
    return maskBool;
}

template <typename T>
std::vector<T> CreateAttentionMaskFromBool(const char* maskBool, const ov::Shape& shape) {
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
                                  bool is_causal,
                                  const Shape& query_shape,
                                  const Shape& key_shape,
                                  const Shape& value_shape,
                                  const Shape& mask_shape,
                                  const Shape& output_shape) {
    std::cout << "{REF}: scaled_dot_product_attention" << std::endl;
    static_assert(std::is_same<T, TMask>::value || std::is_same<TMask, char>::value,
                  "T and TMask must be either the same type, or the TMask must be char(ov::element::boolean)");

    const T* bias = nullptr;
    Shape biasShape = {};

    std::vector<T> attentionMaskData;
    if (mask && !is_causal) {
        biasShape = mask_shape;
        if (std::is_same<TMask, char>::value) {
            attentionMaskData = helpers::CreateAttentionMaskFromBool<T>(mask, biasShape);
            bias = attentionMaskData.data();
        } else {
            bias = helpers::GetRawPtr<T>(mask);
        }
    }

    if (is_causal) {
        auto L = query_shape[query_shape.size() - 2];
        auto S = key_shape[key_shape.size() - 2];
        biasShape = {L, S};
        attentionMaskData = helpers::CreateAttentionMask<T>(biasShape);
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