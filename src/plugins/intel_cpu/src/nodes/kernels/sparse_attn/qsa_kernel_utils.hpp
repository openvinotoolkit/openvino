// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The row arithmetic and the block selection of the QSAIndexer and PagedQSAIndexer kernels, on the simd layer
// (nodes/kernels/simd). It is included by cross-compiled sources only, so the functions are compiled once per
// instruction set (AVX-512, AVX2, plain C++) in the namespace XARCH.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"

namespace ov::Extensions::Cpu::XARCH::qsa_kernel {

// The vector type of a loop body: the full-width one in the main loop, the scalar one in the tail.
template <typename Lanes>
using vec_of = simd::f32_t<Lanes::isa_tag::value>;

// ---- rows of the summary / key / query tensors: f32, f16 or bf16 in memory, f32 in the registers ----

// Read-only f32 view of a row of `n` values: the row itself for f32, otherwise the row converted into `buf`.
template <typename T>
const float* row_as_f32(const T* src, size_t n, float* buf) {
    if constexpr (std::is_same_v<T, float>) {
        return src;
    } else {
        cvt_copy(buf, src, 1, n, 0, 0);
        return buf;
    }
}

// dst[0, n) = src[0, n) rounded to T.
template <typename T>
void store_row(const float* src, T* dst, size_t n) {
    cvt_copy(dst, src, 1, n, 0, 0);
}

// ---- row arithmetic ----

// sum of x[i]^2
inline float sum_squares(const float* x, size_t n) {
    return simd::simd_loop_reduce<2>(
        static_cast<int>(n),
        [&](int j, simd::f32& acc) {
            const auto v = simd::load<simd::f32>(x + j);
            acc = fmadd(v, v, acc);
        },
        [&](int j, float& tail) {
            tail += x[j] * x[j];
        });
}

// dst[i] += src[i]
inline void add_inplace(float* dst, const float* src, size_t n) {
    simd::simd_loop(static_cast<int>(n), [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        store(simd::load<V>(dst + j) + simd::load<V>(src + j), dst + j);
    });
}

// x[i] *= factor
inline void scale_inplace(float* x, float factor, size_t n) {
    simd::simd_loop(static_cast<int>(n), [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        store(simd::load<V>(x + j) * V(factor), x + j);
    });
}

// One row of D = `dim` channels in, one row out (src [dim], weight [dim], cos / sin [rotary_dim], dst [dim]):
//   1. RMSNorm   y = src / sqrt(mean(src^2) + eps) * weight
//   2. RoPE      the split-half (rotate-half) RoPE of HF Transformers over the first `rotary_dim` channels of y:
//                y_rot = y * cos + rotate_half(y) * sin, with rotate_half(y) = concat(-y[half:], y[:half]);
//                the remaining dim - rotary_dim channels are kept as they are.
// cos / sin are already duplicated across the two halves (new_spec/qsa-prepare-17.rst). `dst` may be `src`.
inline void rms_rope(const float* src,
                     float* dst,
                     const float* weight,
                     size_t dim,
                     size_t rotary_dim,
                     float eps,
                     const float* cos,
                     const float* sin) {
    const float inv_rms = 1.0F / std::sqrt(sum_squares(src, dim) / static_cast<float>(dim) + eps);
    simd::simd_loop(static_cast<int>(dim), [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        store(simd::load<V>(src + j) * V(inv_rms) * simd::load<V>(weight + j), dst + j);
    });
    // the pair of channels (j, j + half) is rotated by the angle of the channel j
    const auto half = static_cast<int>(rotary_dim / 2);
    simd::simd_loop(half, [&](int j, auto lanes) {
        using V = vec_of<decltype(lanes)>;
        const V x0 = simd::load<V>(dst + j);
        const V x1 = simd::load<V>(dst + j + half);
        store(x0 * simd::load<V>(cos + j) - x1 * simd::load<V>(sin + j), dst + j);
        store(x1 * simd::load<V>(cos + j + half) + x0 * simd::load<V>(sin + j + half), dst + j + half);
    });
}

// ---- block scores ----

// The rows the scoring handles together: each loaded chunk of a query head is used by all of them, and their
// accumulators are independent chains (the FMA latency would otherwise bound the loop).
inline constexpr size_t kScoreRows = 4;

// scores[i] += sum over the G query heads h of max(0, q[h] . row[i]) for R rows of D f32 channels (`row[r]` is the
// row r), q [G, D] f32.
template <size_t R>
void score_rows(const float* q, const float* const* row, size_t G, size_t D, float* scores) {
    constexpr auto W = static_cast<size_t>(simd::f32::width);
    float sum[R] = {};
    for (size_t h = 0; h < G; h++) {
        const float* const qh = q + h * D;
        simd::f32 acc[R][2];
        size_t d = 0;
        for (; d + 2 * W <= D; d += 2 * W) {
            const auto q0 = simd::load<simd::f32>(qh + d);
            const auto q1 = simd::load<simd::f32>(qh + d + W);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = fmadd(q0, simd::load<simd::f32>(row[r] + d), acc[r][0]);
                acc[r][1] = fmadd(q1, simd::load<simd::f32>(row[r] + d + W), acc[r][1]);
            }
        }
        for (; d + W <= D; d += W) {
            const auto q0 = simd::load<simd::f32>(qh + d);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = fmadd(q0, simd::load<simd::f32>(row[r] + d), acc[r][0]);
            }
        }
        float tail[R] = {};
        for (; d < D; d++) {
            for (size_t r = 0; r < R; r++) {
                tail[r] += qh[d] * row[r][d];
            }
        }
        for (size_t r = 0; r < R; r++) {
            sum[r] += std::max(0.0F, reduce(acc[r][0] + acc[r][1]) + tail[r]);
        }
    }
    for (size_t r = 0; r < R; r++) {
        scores[r] += sum[r];
    }
}

// scores[i] += sum over the G query heads h of max(0, q[h] . rows[i]), for the n summary rows ([n, D] of the type T)
// of one index key head and its G query heads q ([G, D] f32). The rows are converted to f32 once (they are shared by
// the heads) into `stage`, kScoreRows * D floats (not touched for f32 rows).
template <typename T>
void score_run(const float* q, const T* rows, size_t n, size_t G, size_t D, float* scores, float* stage) {
    size_t i = 0;
    for (; i + kScoreRows <= n; i += kScoreRows) {
        const float* row[kScoreRows];
        for (size_t r = 0; r < kScoreRows; r++) {
            row[r] = row_as_f32(rows + (i + r) * D, D, stage + r * D);
        }
        score_rows<kScoreRows>(q, row, G, D, scores + i);
    }
    for (; i < n; i++) {
        const float* const row = row_as_f32(rows + i * D, D, stage);
        score_rows<1>(q, &row, G, D, scores + i);
    }
}

// ---- block selection ----

// The (score, block id) of a complete block.
using BlockScore = std::pair<float, int32_t>;

// Writes the ids of the k = min(block_topk, num_blocks) blocks of the highest scores[0, num_blocks) to `chosen` and
// returns k. Equal scores: the lower block id wins. The blocks are in ascending id order, or in descending score order
// (equal scores: the lower id first) with order_by_score. `scores` is reordered.
inline size_t select_topk(BlockScore* scores,
                          int32_t* chosen,
                          size_t num_blocks,
                          size_t block_topk,
                          bool order_by_score) {
    const size_t k = std::min(block_topk, num_blocks);
    std::partial_sort(scores, scores + k, scores + num_blocks, [](const BlockScore& lhs, const BlockScore& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first > rhs.first;
        }
        return lhs.second < rhs.second;
    });
    for (size_t i = 0; i < k; i++) {
        chosen[i] = scores[i].second;
    }
    if (!order_by_score) {
        std::sort(chosen, chosen + k);
    }
    return k;
}

}  // namespace ov::Extensions::Cpu::XARCH::qsa_kernel
