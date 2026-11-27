// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test {

// The steps of the query that the op applies: the query_norm / query_rope attributes.
enum class QSAIndexerQueryFusionType { kNormAndRope, kNormOnly, kRopeOnly, kNone };

// float <-> f32 / f16 / bf16 tensors.
inline std::vector<float> read_tensor_to_f32(const ov::Tensor& t) {
    std::vector<float> out(t.get_size());
    if (t.get_element_type() == ov::element::bf16) {
        std::copy_n(t.data<const ov::bfloat16>(), out.size(), out.begin());
    } else if (t.get_element_type() == ov::element::f16) {
        std::copy_n(t.data<const ov::float16>(), out.size(), out.begin());
    } else {
        std::copy_n(t.data<const float>(), out.size(), out.begin());
    }
    return out;
}

inline void write_f32_to_tensor(ov::Tensor& t, const std::vector<float>& data) {
    if (t.get_element_type() == ov::element::bf16) {
        std::copy(data.begin(), data.end(), t.data<ov::bfloat16>());
    } else if (t.get_element_type() == ov::element::f16) {
        std::copy(data.begin(), data.end(), t.data<ov::float16>());
    } else {
        std::copy(data.begin(), data.end(), t.data<float>());
    }
}

// Rotate-half RoPE of the first `rotary_dim` channels of `row`; cos_row / sin_row hold `rotary_dim` values each.
inline void rope_rotate_half(float* row, size_t rotary_dim, const float* cos_row, const float* sin_row) {
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float x0 = row[j];
        const float x1 = row[j + half];
        row[j] = x0 * cos_row[j] - x1 * sin_row[j];
        row[j + half] = x1 * cos_row[j + half] + x0 * sin_row[j + half];
    }
}

// RMSNorm with the weight, then the RoPE of the first `rotary_dim` channels (the steps the op applies: `norm`, `rope`).
inline void query_row_ref(const float* src,
                          float* dst,
                          const float* weight,
                          size_t dim,
                          size_t rotary_dim,
                          float eps,
                          const float* cos_row,
                          const float* sin_row,
                          bool norm,
                          bool rope) {
    if (norm) {
        float sumsq = 0.0F;
        for (size_t d = 0; d < dim; d++) {
            sumsq += src[d] * src[d];
        }
        const float inv_rms = 1.0F / std::sqrt(sumsq / static_cast<float>(dim) + eps);
        for (size_t d = 0; d < dim; d++) {
            dst[d] = src[d] * inv_rms * weight[d];
        }
    } else {
        std::copy_n(src, dim, dst);
    }
    if (rope) {
        rope_rotate_half(dst, rotary_dim, cos_row, sin_row);
    }
}

}  // namespace ov::test
