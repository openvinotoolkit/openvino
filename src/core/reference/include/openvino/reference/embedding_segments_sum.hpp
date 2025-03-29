// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void embeddingSegmentsSum(const T* embTable,
                          const U* indices,
                          const U* segmentIds,
                          const U* defaultIndex,
                          const T* weights,
                          T* out,
                          const Shape& embTableShape,
                          const Shape& indicesShape,
                          const Shape& outShape) {
    const size_t indices_len = indicesShape[0];
    const size_t segments_num = outShape[0];

    size_t embDepth = 1lu;
    for (size_t i = 1; i < outShape.size(); i++) {
        embDepth *= outShape[i];
    }
    std::fill(out, out + shape_size(outShape), T{0});

    bool with_weights = (weights != nullptr);

    for (size_t index = 0; index < indices_len; index++) {
        size_t obi = segmentIds[index];
        if (obi >= segments_num)
            OPENVINO_THROW("Segment index could not be more than segments number");
        size_t dst_index = obi * embDepth;
        size_t src_index = indices[index] * embDepth;

        if (with_weights) {
            for (size_t i = 0lu; i < embDepth; i++) {
                out[dst_index + i] += embTable[src_index + i] * weights[index];
            }
        } else {
            for (size_t i = 0lu; i < embDepth; i++) {
                out[dst_index + i] += embTable[src_index + i];
            }
        }
    }

    if (defaultIndex != nullptr) {
        U defIndex = defaultIndex[0];
        if (defIndex < U(0) && static_cast<size_t>(defIndex) >= embTableShape[0])
            OPENVINO_THROW(std::string("Invalid default index") + std::to_string(defIndex));
        for (size_t obi = 0; obi < segments_num; obi++) {
            bool found = false;
            for (size_t index = 0; index < indices_len; index++) {
                if (static_cast<size_t>(segmentIds[index]) == obi) {
                    found = true;
                    break;
                }
            }
            if (found)
                continue;
            size_t src_index = defIndex * embDepth;
            size_t dst_index = obi * embDepth;
            for (size_t i = 0lu; i < embDepth; i++) {
                out[dst_index + i] = embTable[src_index + i];
            }
        }
    }
}  // embeddingSegmentsSum

}  // namespace reference
}  // namespace ov
