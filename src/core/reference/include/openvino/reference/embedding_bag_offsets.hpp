// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/util/embeddingbag_offsets_base.hpp"
#include "openvino/reference/divide.hpp"

namespace ov {
namespace reference {
template <typename T, typename U>
void embeddingBagOffsets(const T* emb_table,
                         const U* indices,
                         const U* offsets,
                         const U* default_index,
                         const T* weights,
                         T* out,
                         const size_t indices_count,
                         const Shape& outShape,
                         ov::op::util::EmbeddingBagOffsetsBase::Reduction reduction) {
    using Reduction = ov::op::util::EmbeddingBagOffsetsBase::Reduction;
    const size_t offsets_size = outShape[0];
    std::vector<U> default_indices;
    if (default_index)
        default_indices.push_back(default_index[0]);

    size_t embDepth = 1;
    for (size_t i = 1; i < outShape.size(); i++) {
        embDepth *= outShape[i];
    }
    std::fill(out, out + shape_size(outShape), T{0});
    auto get_indices =
        [&](size_t emb_index, const U*& indices_ref, size_t& indices_num, size_t& weights_idx, bool& with_weights) {
            if (emb_index >= offsets_size)
                OPENVINO_THROW("Invalid embedding bag index.");
            if (static_cast<size_t>(offsets[emb_index]) >= indices_count)
                OPENVINO_THROW(std::string("Offset value exceeds indices size in the model.\noffset: ") +
                               std::to_string(offsets[emb_index]) + "; indices size: " + std::to_string(indices_count));

            indices_ref = nullptr;
            indices_num = 0lu;
            with_weights = (weights != nullptr);

            if (emb_index == offsets_size - 1lu)
                indices_num = indices_count - offsets[emb_index];
            else
                indices_num = offsets[emb_index + 1lu] - offsets[emb_index];

            if (indices_num != 0lu) {
                indices_ref = indices + offsets[emb_index];
            } else {
                // Empty or default bag
                with_weights = false;
                if (default_indices.size() == 1lu && *default_indices.data() >= 0) {
                    indices_ref = default_indices.data();
                    indices_num = 1lu;
                }
                return;
            }

            if (with_weights)
                weights_idx = offsets[emb_index];
        };

    size_t indices_size = 0lu;
    const U* indices_emb = nullptr;
    size_t weights_idx = 0lu;
    bool with_weights_b = (weights != nullptr);
    bool with_weights = with_weights_b;

    for (size_t obi = 0lu; obi < outShape.at(0); obi++) {
        size_t dst_index = obi * embDepth;
        get_indices(obi, indices_emb, indices_size, weights_idx, with_weights);
        if (indices_emb != nullptr) {
            with_weights = with_weights_b & with_weights;
            for (size_t in_idx = 0lu; in_idx < indices_size; in_idx++) {
                size_t src_index = indices_emb[in_idx] * embDepth;

                if (with_weights) {
                    for (size_t i = 0lu; i < embDepth; i++) {
                        out[dst_index + i] += emb_table[src_index + i] * weights[weights_idx];
                    }
                    weights_idx++;
                } else {
                    for (size_t i = 0lu; i < embDepth; i++) {
                        out[dst_index + i] += emb_table[src_index + i];
                    }
                }
            }
            if (reduction == Reduction::MEAN) {
                for (size_t i = 0lu; i < embDepth; i++) {
                    out[dst_index + i] /= (T)indices_size;
                }
            }
        }
    }

}  // embeddingBagOffsetsSum

}  // namespace reference
}  // namespace ov
