// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// The row arithmetic and the block selection of the QSAIndexer and PagedQSAIndexer kernels. It is included by
// cross-compiled sources only, so the functions are compiled once per instruction set (AVX-512, AVX2, plain C++) in
// the namespace XARCH. One source serves all of them: `vfloat` is a vector of `kLanes` floats (16 / 8 / 1), and the
// same loops run on every set.

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH::qsa {

#if defined(HAVE_AVX512F)

using vfloat = __m512;
inline constexpr size_t kLanes = 16;
inline vfloat vzero() {
    return _mm512_setzero_ps();
}
inline vfloat vset1(float x) {
    return _mm512_set1_ps(x);
}
inline vfloat vload(const float* p) {
    return _mm512_loadu_ps(p);
}
inline void vstore(float* p, vfloat v) {
    _mm512_storeu_ps(p, v);
}
inline vfloat vadd(vfloat a, vfloat b) {
    return _mm512_add_ps(a, b);
}
inline vfloat vsub(vfloat a, vfloat b) {
    return _mm512_sub_ps(a, b);
}
inline vfloat vmul(vfloat a, vfloat b) {
    return _mm512_mul_ps(a, b);
}
inline vfloat vfma(vfloat a, vfloat b, vfloat c) {  // a * b + c
    return _mm512_fmadd_ps(a, b, c);
}
inline float vsum(vfloat v) {
    return _mm512_reduce_add_ps(v);
}
// The first n < kLanes lanes of a row; the other lanes are zero / not stored.
inline vfloat vload_tail(const float* p, size_t n) {
    return _mm512_maskz_loadu_ps(static_cast<__mmask16>((1U << n) - 1U), p);
}
inline void vstore_tail(float* p, vfloat v, size_t n) {
    _mm512_mask_storeu_ps(p, static_cast<__mmask16>((1U << n) - 1U), v);
}
// Selection primitives: lanewise minimum / maximum, their reduction, comparison masks (bit l = lane l), the bucket of
// each score (see select_top_blocks) and the compaction of the lanes of a mask.
inline vfloat vmin(vfloat a, vfloat b) {
    return _mm512_min_ps(a, b);
}
inline vfloat vmax(vfloat a, vfloat b) {
    return _mm512_max_ps(a, b);
}
inline float vreduce_min(vfloat v) {
    return _mm512_reduce_min_ps(v);
}
inline float vreduce_max(vfloat v) {
    return _mm512_reduce_max_ps(v);
}
inline uint32_t vcmp_gt(vfloat a, vfloat b) {
    return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
}
inline uint32_t vcmp_eq(vfloat a, vfloat b) {
    return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
}
// int(min(max_bucket, trunc((v - low) * to_bucket))) of every lane
inline __m512i vbucket(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket) {
    const __m512i bucket = _mm512_cvttps_epi32(_mm512_mul_ps(_mm512_sub_ps(v, low), to_bucket));
    return _mm512_min_epi32(bucket, _mm512_set1_epi32(max_bucket));
}
inline void vstore_buckets(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t* out) {
    _mm512_storeu_si512(out, vbucket(v, low, to_bucket, max_bucket));
}
inline uint32_t vbucket_mask(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t wanted) {
    return _mm512_cmpeq_epi32_mask(vbucket(v, low, to_bucket, max_bucket), _mm512_set1_epi32(wanted));
}
// Stores the lanes of `v` selected by `mask` one after the other to dst (nothing else is written), returns how many.
inline size_t vcompress_store(float* dst, vfloat v, uint32_t mask) {
    _mm512_mask_compressstoreu_ps(dst, static_cast<__mmask16>(mask), v);
    return std::bitset<32>(mask).count();
}
// The same for the ids base, base + 1, ... of the lanes.
inline size_t vcompress_ids(int32_t* dst, int32_t base, uint32_t mask) {
    const __m512i ids = _mm512_add_epi32(_mm512_set1_epi32(base),
                                         _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    _mm512_mask_compressstoreu_epi32(dst, static_cast<__mmask16>(mask), ids);
    return std::bitset<32>(mask).count();
}

#elif defined(HAVE_AVX2)

using vfloat = __m256;
inline constexpr size_t kLanes = 8;
inline vfloat vzero() {
    return _mm256_setzero_ps();
}
inline vfloat vset1(float x) {
    return _mm256_set1_ps(x);
}
inline vfloat vload(const float* p) {
    return _mm256_loadu_ps(p);
}
inline void vstore(float* p, vfloat v) {
    _mm256_storeu_ps(p, v);
}
inline vfloat vadd(vfloat a, vfloat b) {
    return _mm256_add_ps(a, b);
}
inline vfloat vsub(vfloat a, vfloat b) {
    return _mm256_sub_ps(a, b);
}
inline vfloat vmul(vfloat a, vfloat b) {
    return _mm256_mul_ps(a, b);
}
inline vfloat vfma(vfloat a, vfloat b, vfloat c) {  // a * b + c
    return _mm256_fmadd_ps(a, b, c);
}
inline float vsum(vfloat v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
inline vfloat vload_tail(const float* p, size_t n) {
    return _mm256_maskload_ps(p, get_mask(static_cast<int>(n)));
}
inline void vstore_tail(float* p, vfloat v, size_t n) {
    _mm256_maskstore_ps(p, get_mask(static_cast<int>(n)), v);
}
// Selection primitives: lanewise minimum / maximum, their reduction, comparison masks (bit l = lane l), the bucket of
// each score (see select_top_blocks) and the compaction of the lanes of a mask.
inline vfloat vmin(vfloat a, vfloat b) {
    return _mm256_min_ps(a, b);
}
inline vfloat vmax(vfloat a, vfloat b) {
    return _mm256_max_ps(a, b);
}
inline float vreduce_min(vfloat v) {
    __m128 m = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_min_ps(m, _mm_movehl_ps(m, m));
    m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}
inline float vreduce_max(vfloat v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}
inline uint32_t vcmp_gt(vfloat a, vfloat b) {
    return static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_GT_OQ)));
}
inline uint32_t vcmp_eq(vfloat a, vfloat b) {
    return static_cast<uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ)));
}
// int(min(max_bucket, trunc((v - low) * to_bucket))) of every lane
inline __m256i vbucket(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket) {
    const __m256i bucket = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, low), to_bucket));
    return _mm256_min_epi32(bucket, _mm256_set1_epi32(max_bucket));
}
inline void vstore_buckets(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t* out) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), vbucket(v, low, to_bucket, max_bucket));
}
inline uint32_t vbucket_mask(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t wanted) {
    const __m256i equal = _mm256_cmpeq_epi32(vbucket(v, low, to_bucket, max_bucket), _mm256_set1_epi32(wanted));
    return static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(equal)));
}

#else

using vfloat = float;
inline constexpr size_t kLanes = 1;
inline vfloat vzero() {
    return 0.0F;
}
inline vfloat vset1(float x) {
    return x;
}
inline vfloat vload(const float* p) {
    return *p;
}
inline void vstore(float* p, vfloat v) {
    *p = v;
}
inline vfloat vadd(vfloat a, vfloat b) {
    return a + b;
}
inline vfloat vsub(vfloat a, vfloat b) {
    return a - b;
}
inline vfloat vmul(vfloat a, vfloat b) {
    return a * b;
}
inline vfloat vfma(vfloat a, vfloat b, vfloat c) {
    return a * b + c;
}
inline float vsum(vfloat v) {
    return v;
}
// A single lane has no tail: these are never reached.
inline vfloat vload_tail(const float* p, size_t /*n*/) {
    return *p;
}
inline void vstore_tail(float* p, vfloat v, size_t /*n*/) {
    *p = v;
}
// Selection primitives: lanewise minimum / maximum, their reduction, comparison masks (bit l = lane l), the bucket of
// each score (see select_top_blocks) and the compaction of the lanes of a mask.
inline vfloat vmin(vfloat a, vfloat b) {
    return std::min(a, b);
}
inline vfloat vmax(vfloat a, vfloat b) {
    return std::max(a, b);
}
inline float vreduce_min(vfloat v) {
    return v;
}
inline float vreduce_max(vfloat v) {
    return v;
}
inline uint32_t vcmp_gt(vfloat a, vfloat b) {
    return a > b ? 1U : 0U;
}
inline uint32_t vcmp_eq(vfloat a, vfloat b) {
    return a == b ? 1U : 0U;
}
inline int32_t vbucket_one(float v, float low, float to_bucket, int32_t max_bucket) {
    return std::min(max_bucket, static_cast<int32_t>((v - low) * to_bucket));
}
inline void vstore_buckets(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t* out) {
    out[0] = vbucket_one(v, low, to_bucket, max_bucket);
}
inline uint32_t vbucket_mask(vfloat v, vfloat low, vfloat to_bucket, int32_t max_bucket, int32_t wanted) {
    return vbucket_one(v, low, to_bucket, max_bucket) == wanted ? 1U : 0U;
}

#endif

#if !defined(HAVE_AVX512F)
// Compaction without a compress instruction: every lane is stored at the write position, which advances only for the
// selected ones (branch-free; the position past the last selected lane is written too, so dst needs one spare slot).
inline size_t vcompress_store(float* dst, vfloat v, uint32_t mask) {
    alignas(64) float lanes[kLanes];
    vstore(lanes, v);
    size_t count = 0;
    for (size_t lane = 0; lane < kLanes; lane++) {
        dst[count] = lanes[lane];
        count += (mask >> lane) & 1U;
    }
    return count;
}
inline size_t vcompress_ids(int32_t* dst, int32_t base, uint32_t mask) {
    size_t count = 0;
    for (size_t lane = 0; lane < kLanes; lane++) {
        dst[count] = base + static_cast<int32_t>(lane);
        count += (mask >> lane) & 1U;
    }
    return count;
}
#endif

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
    vfloat acc0 = vzero();
    vfloat acc1 = vzero();
    size_t i = 0;
    for (; i + 2 * kLanes <= n; i += 2 * kLanes) {
        const vfloat v0 = vload(x + i);
        const vfloat v1 = vload(x + i + kLanes);
        acc0 = vfma(v0, v0, acc0);
        acc1 = vfma(v1, v1, acc1);
    }
    if (i + kLanes <= n) {
        const vfloat v0 = vload(x + i);
        acc0 = vfma(v0, v0, acc0);
        i += kLanes;
    }
    if (i < n) {
        const vfloat v0 = vload_tail(x + i, n - i);
        acc1 = vfma(v0, v0, acc1);
    }
    return vsum(vadd(acc0, acc1));
}

// dst[i] += src[i]
inline void add_inplace(float* dst, const float* src, size_t n) {
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        vstore(dst + i, vadd(vload(dst + i), vload(src + i)));
    }
    if (i < n) {
        vstore_tail(dst + i, vadd(vload_tail(dst + i, n - i), vload_tail(src + i, n - i)), n - i);
    }
}

// x[i] *= factor
inline void scale_inplace(float* x, float factor, size_t n) {
    const vfloat vfactor = vset1(factor);
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        vstore(x + i, vmul(vload(x + i), vfactor));
    }
    if (i < n) {
        vstore_tail(x + i, vmul(vload_tail(x + i, n - i), vfactor), n - i);
    }
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
    const vfloat vinv = vset1(inv_rms);
    size_t d = 0;
    for (; d + kLanes <= dim; d += kLanes) {
        vstore(dst + d, vmul(vmul(vload(src + d), vinv), vload(weight + d)));
    }
    if (d < dim) {
        const size_t rest = dim - d;
        vstore_tail(dst + d, vmul(vmul(vload_tail(src + d, rest), vinv), vload_tail(weight + d, rest)), rest);
    }

    // the pair of channels (j, j + half) is rotated by the angle of the channel j
    const size_t half = rotary_dim / 2;
    size_t j = 0;
    for (; j + kLanes <= half; j += kLanes) {
        const vfloat x0 = vload(dst + j);
        const vfloat x1 = vload(dst + j + half);
        vstore(dst + j, vsub(vmul(x0, vload(cos + j)), vmul(x1, vload(sin + j))));
        vstore(dst + j + half, vadd(vmul(x1, vload(cos + j + half)), vmul(x0, vload(sin + j + half))));
    }
    if (j < half) {
        const size_t rest = half - j;
        const vfloat x0 = vload_tail(dst + j, rest);
        const vfloat x1 = vload_tail(dst + j + half, rest);
        vstore_tail(dst + j, vsub(vmul(x0, vload_tail(cos + j, rest)), vmul(x1, vload_tail(sin + j, rest))), rest);
        vstore_tail(dst + j + half,
                    vadd(vmul(x1, vload_tail(cos + j + half, rest)), vmul(x0, vload_tail(sin + j + half, rest))),
                    rest);
    }
}

// ---- block scores ----

// The rows the scoring handles together: each loaded chunk of a query head is used by all of them, and their
// accumulators are independent chains (the FMA latency would otherwise bound the loop).
inline constexpr size_t kScoreRows = 4;

// scores[i] += sum over the G query heads h of max(0, q[h] . row[i]) for R rows of D f32 channels (`row[r]` is the
// row r), q [G, D] f32.
template <size_t R>
void score_rows(const float* q, const float* const* row, size_t G, size_t D, float* scores) {
    float sum[R] = {};
    for (size_t h = 0; h < G; h++) {
        const float* const qh = q + h * D;
        vfloat acc[R][2];
        for (size_t r = 0; r < R; r++) {
            acc[r][0] = vzero();
            acc[r][1] = vzero();
        }
        size_t d = 0;
        for (; d + 2 * kLanes <= D; d += 2 * kLanes) {
            const vfloat q0 = vload(qh + d);
            const vfloat q1 = vload(qh + d + kLanes);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = vfma(q0, vload(row[r] + d), acc[r][0]);
                acc[r][1] = vfma(q1, vload(row[r] + d + kLanes), acc[r][1]);
            }
        }
        for (; d + kLanes <= D; d += kLanes) {
            const vfloat q0 = vload(qh + d);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = vfma(q0, vload(row[r] + d), acc[r][0]);
            }
        }
        if (d < D) {
            const vfloat q0 = vload_tail(qh + d, D - d);
            for (size_t r = 0; r < R; r++) {
                acc[r][0] = vfma(q0, vload_tail(row[r] + d, D - d), acc[r][0]);
            }
        }
        for (size_t r = 0; r < R; r++) {
            sum[r] += std::max(0.0F, vsum(vadd(acc[r][0], acc[r][1])));
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

}  // namespace ov::Extensions::Cpu::XARCH::qsa
