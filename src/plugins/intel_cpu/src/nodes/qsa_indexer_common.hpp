// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cmath>
#include <cstddef>

// The math the dense QSAIndexer and the PagedQSAIndexer nodes share: the cache layouts and the indexing differ, the
// per-row RMSNorm + partial RoPE does not.
namespace ov::intel_cpu::node::qsa {

inline float rms_inv_scale(const float* x, size_t n, float eps) {
    float sumsq = 0.0F;
    for (size_t i = 0; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

// RMSNorm(src) * weight, followed by the split-half (rotate-half) RoPE of HF Transformers over the first `rotary_dim`
// channels: x * cos + rotate_half(x) * sin, with rotate_half(x) = concat(-x[half:], x[:half]). The coefficient of the
// rotary channel d is cos[d * cos_sin_stride] / sin[d * cos_sin_stride]; the dense node passes two arrays of
// `rotary_dim` values (stride 1; the cos / sin are already duplicated across the two halves, per
// new_spec/qsa-prepare-17.rst), the paged node one interleaved (cos_d, sin_d) row (cos = row, sin = row + 1, stride 2).
// Writes `dim` channels to `dst` (may alias `src`).
inline void rmsnorm_partial_rope(const float* src,
                                 float* dst,
                                 const float* weight,
                                 size_t dim,
                                 size_t rotary_dim,
                                 float eps,
                                 const float* cos,
                                 const float* sin,
                                 size_t cos_sin_stride = 1) {
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    const size_t half = rotary_dim / 2;
    for (size_t j = 0; j < half; j++) {
        const float cos_lo = cos[j * cos_sin_stride];
        const float sin_lo = sin[j * cos_sin_stride];
        const float cos_hi = cos[(j + half) * cos_sin_stride];
        const float sin_hi = sin[(j + half) * cos_sin_stride];
        const float x0 = dst[j];
        const float x1 = dst[j + half];
        dst[j] = x0 * cos_lo - x1 * sin_lo;
        dst[j + half] = x1 * cos_hi + x0 * sin_hi;
    }
}

}  // namespace ov::intel_cpu::node::qsa
