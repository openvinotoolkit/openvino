// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "nodes/common/cpu_convert.h"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

// The math the dense QSAIndexer and the PagedQSAIndexer nodes share: the cache layouts and the indexing differ, the
// per-row RMSNorm + partial RoPE does not.
namespace ov::intel_cpu::node::qsa {

// What one row of the batch contributes to a call of the dense QSAIndexer. Every quantity is derived from the row's
// own key_length and the window length L of the call alone -- no cross-call running counter is kept.
struct RowPlan {
    int64_t num_padding;  // left-padded positions of this call's window, skipped everywhere
    int64_t current;      // real tokens of this call
    int64_t tail;         // rows already sitting in the incomplete-block state
    int64_t past_blocks;  // complete blocks of the earlier calls
    int64_t new_blocks;   // blocks this call completes

    // Complete blocks visible to the token `v` of this call (`v` counts the real tokens, left padding excluded).
    [[nodiscard]] int64_t blocks_visible(int64_t v, int64_t ratio) const {
        return past_blocks + (tail + v + 1) / ratio;
    }
};

inline RowPlan make_row_plan(int64_t key_length, int64_t window, int64_t ratio) {
    const int64_t past = std::max<int64_t>(0, key_length - window);
    RowPlan plan{};
    plan.current = key_length - past;  // == min(window, key_length)
    plan.num_padding = window - plan.current;
    plan.tail = past % ratio;
    plan.past_blocks = past / ratio;
    plan.new_blocks = (plan.tail + plan.current) / ratio;
    return plan;
}

// Converts a short row to / from f32. cpu_convert dispatches on every call, which costs more than the conversion of
// a row of a few hundred values: bf16 is converted by a plain loop.
template <typename T>
void row_to_f32(const T* src, float* dst, size_t len) {
    if constexpr (std::is_same_v<T, ov::bfloat16>) {
        for (size_t i = 0; i < len; i++) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        cpu_convert(src, dst, ov::element::from<T>(), ov::element::f32, len);
    }
}

template <typename T>
void row_from_f32(const float* src, T* dst, size_t len) {
    if constexpr (std::is_same_v<T, float>) {
        std::copy_n(src, len, dst);
    } else if constexpr (std::is_same_v<T, ov::bfloat16>) {
        for (size_t i = 0; i < len; i++) {
            dst[i] = ov::bfloat16(src[i]);
        }
    } else {
        cpu_convert(src, dst, ov::element::f32, ov::element::from<T>(), len);
    }
}

// All floating-point ports of the dense node share the precision T (the node declares them that way), so only the rows
// actually touched are converted to / from the f32 compute buffers, never a whole tensor.
// Read-only f32 view of a row: the tensor row itself when T is f32 (nothing to convert, so no copy), otherwise the
// row converted into `scratch`.
template <typename T, typename... Idx>
const float* row_f32(const PlainTensor& t, size_t len, float* scratch, Idx... idx) {
    if constexpr (std::is_same_v<T, float>) {
        return t.ptr<float>(idx...);
    } else {
        row_to_f32(t.ptr<T>(idx...), scratch, len);
        return scratch;
    }
}

inline float rms_inv_scale(const float* x, size_t n, float eps) {
    // Independent partial sums: a single accumulator is a serial chain the compiler may not break up (no fast-math)
    constexpr size_t lanes = 8;
    float partial[lanes] = {};
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        for (size_t lane = 0; lane < lanes; lane++) {
            partial[lane] += x[i + lane] * x[i + lane];
        }
    }
    float sumsq = 0.0F;
    for (; i < n; i++) {
        sumsq += x[i] * x[i];
    }
    for (const float lane_sum : partial) {
        sumsq += lane_sum;
    }
    return 1.0F / std::sqrt(sumsq / static_cast<float>(n) + eps);
}

// One row of D = `dim` channels in, one row out (shapes: src [dim], weight [dim], cos / sin [rotary_dim], dst [dim]):
//   1. RMSNorm   y = src / sqrt(mean(src^2) + eps) * weight
//   2. RoPE      the split-half (rotate-half) RoPE of HF Transformers over the first `rotary_dim` channels of y:
//                y_rot = y * cos + rotate_half(y) * sin, with rotate_half(y) = concat(-y[half:], y[:half]);
//                the remaining dim - rotary_dim channels are kept as they are.
// The coefficient of the rotary channel d is cos[d * cos_sin_stride] / sin[d * cos_sin_stride]; the dense node passes
// two arrays of `rotary_dim` values (stride 1; the cos / sin are already duplicated across the two halves, per
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
    // step 1: RMSNorm
    const float inv_rms = rms_inv_scale(src, dim, eps);
    for (size_t d = 0; d < dim; d++) {
        dst[d] = src[d] * inv_rms * weight[d];
    }
    // step 2: RoPE, the pair of channels (j, j + half) is rotated by the angle of the channel j
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
