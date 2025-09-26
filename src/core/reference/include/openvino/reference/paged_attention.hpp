// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"

// Cache manager API
#include "openvino/core/cache_manager.hpp"

#ifndef PA_DEBUG
#    define PA_DEBUG 0
#endif

namespace ov::reference {

// ============================================================================
// Small debug helpers (gated by PA_DEBUG)
// ============================================================================
namespace pa_debug {

template <typename T>
void PrintVector(const T* data, size_t length, const char* label) {
#if PA_DEBUG
    std::printf("%s [", label);
    for (size_t i = 0; i < length; ++i) {
        std::printf("%s%.6f", (i ? ", " : ""), static_cast<double>(data[i]));
    }
    std::printf("]\n");
#else
    (void)data;
    (void)length;
    (void)label;
#endif
}

inline void PrintI32(int32_t v, const char* label) {
#if PA_DEBUG
    std::printf("%s %d\n", label, v);
#else
    (void)v;
    (void)label;
#endif
}

inline void PrintSize(size_t v, const char* label) {
#if PA_DEBUG
    std::printf("%s %zu\n", label, v);
#else
    (void)v;
    (void)label;
#endif
}

template <typename T>
void PrintScalar(T v, const char* label) {
#if PA_DEBUG
    std::printf("%s %.6f\n", label, static_cast<double>(v));
#else
    (void)v;
    (void)label;
#endif
}

}  // namespace pa_debug

// ============================================================================
// Math utilities
// ============================================================================
namespace pa_math {

template <typename T>
inline T ComputeDotProduct(const T* a, const T* b, size_t length) {
    T acc = T(0);
    for (size_t i = 0; i < length; ++i)
        acc += a[i] * b[i];
    return acc;
}

template <typename T>
inline void ComputeSoftmaxInPlace(std::vector<T>& values) {
    const T max_value = *std::max_element(values.begin(), values.end());
    T sum = T(0);
    for (T& v : values) {
        v = std::exp(v - max_value);
        sum += v;
    }
    const T inv_sum = sum ? T(1) / sum : T(0);
    for (T& v : values)
        v *= inv_sum;
}

}  // namespace pa_math

// ============================================================================
// Rotary helpers
// ============================================================================
namespace pa_rotary {

template <typename T>
inline void ApplyRotaryEmbeddingToVector(T* vec, size_t head_size, const T* trig_lut, size_t trig_index) {
    const size_t half = head_size / 2;
    const T* row = trig_lut + trig_index * head_size;
    for (size_t i = 0; i < half; ++i) {
        const T x0 = vec[2 * i];
        const T x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * row[i] - x1 * row[half + i];
        vec[2 * i + 1] = x0 * row[half + i] + x1 * row[i];
    }
}

inline bool BlockHasRotary(int32_t block_id,
                           const int32_t* rotated_block_indices,
                           size_t num_rotated_blocks,
                           int32_t& out_rotated_index) {
    if (!rotated_block_indices)
        return false;
    for (size_t i = 0; i < num_rotated_blocks; ++i) {
        if (rotated_block_indices[i] == block_id) {
            out_rotated_index = static_cast<int32_t>(i);
            return true;
        }
    }
    return false;
}

inline int32_t ComputeTrigRowIndex(int32_t rotated_index,
                                   int32_t token_offset_in_block,
                                   const int32_t* rotation_deltas,
                                   size_t rotation_deltas_dim,
                                   size_t block_size) {
    if (!rotation_deltas)
        return 0;
    if (rotation_deltas_dim == 1)
        return rotation_deltas[rotated_index];
    return rotation_deltas[rotated_index * static_cast<int32_t>(block_size) + token_offset_in_block];
}

}  // namespace pa_rotary

// ============================================================================
// Token/sequence helpers
// ============================================================================
inline int32_t FindFirstNegativeIndex(const int32_t* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (data[i] < 0)
            return static_cast<int32_t>(i);
    }
    return -1;
}

inline size_t ResolveSequenceIndexForToken(size_t token_index,
                                           const int32_t* subsequence_begins,
                                           size_t sequence_count) {
    if (!subsequence_begins || sequence_count <= 1)
        return 0;
    for (size_t s = 0; s < sequence_count; ++s) {
        if (token_index >= static_cast<size_t>(subsequence_begins[s]) &&
            token_index < static_cast<size_t>(subsequence_begins[s + 1])) {
            return s;
        }
    }
    return 0;
}

inline int32_t AcquireOrRecycleBlockForSequence(size_t sequence_index,
                                                int32_t* block_indices,
                                                int32_t* block_indices_begins,
                                                std::vector<int32_t>& sequence_block_count,
                                                size_t max_blocks) {
    const int32_t begin = block_indices_begins[sequence_index];
    const int32_t count = sequence_block_count[sequence_index];

    if (count < std::numeric_limits<int32_t>::max()) {
        const int32_t free_block = FindFirstNegativeIndex(block_indices, max_blocks);
        if (free_block != -1) {
            sequence_block_count[sequence_index] = count + 1;
            return free_block;
        }
    }

    const int32_t oldest = block_indices[begin];
    for (int32_t i = 0; i + 1 < count; ++i) {
        block_indices[begin + i] = block_indices[begin + i + 1];
    }
    return oldest;
}

// ============================================================================
// Compact context assembled from CacheManager and call-site inputs
// ============================================================================
struct PagedAttentionKernelContext {
    const void* query{nullptr};
    const void* key{nullptr};
    const void* value{nullptr};

    void* key_cache_base{nullptr};
    void* value_cache_base{nullptr};

    const int32_t* past_lens{nullptr};
    const int32_t* subsequence_begins{nullptr};

    int32_t* block_indices{nullptr};
    int32_t* block_indices_begins{nullptr};

    const void* alibi_slopes{nullptr};

    const int32_t* rotated_block_indices{nullptr};
    const int32_t* rotation_deltas{nullptr};
    const void* rotation_trig_lut{nullptr};

    std::vector<int32_t> sequence_block_count;

    size_t batch_token_count{0};
    size_t sequence_count{0};
    size_t head_count{0};
    size_t block_size{0};
    size_t block_count{0};
    size_t query_head_size{0};
    size_t key_head_size{0};
    size_t value_head_size{0};
    size_t query_feature_size{0};
    size_t key_feature_size{0};
    size_t value_feature_size{0};

    int32_t max_context_length{0};
    int32_t sliding_window{0};

    size_t rotated_block_count{0};
    size_t rotation_lut_rows{0};
    size_t rotation_deltas_dim{0};
};

// ============================================================================
// CacheManager adapter
// ============================================================================
struct CacheManagerAdapter {
    ov::internal::CacheManager& cache_manager;
    ov::internal::CacheManager::Handle handle;

    explicit CacheManagerAdapter(ov::internal::CacheManager& cm, ov::internal::CacheManager::Handle h)
        : cache_manager(cm),
          handle(h) {}

    inline void* GetKeyCacheBase() const {
        return cache_manager.get_cache_blocks().key_base;
    }
    inline void* GetValueCacheBase() const {
        return cache_manager.get_cache_blocks().value_base;
    }

    inline const int32_t* GetSubsequenceBeginsOrNull() const {
        auto sv = cache_manager.get_subsequence_begins(handle);
        return sv.data;
    }

    // Layout is owned by CacheManager and captured during operator registration.
    struct OperatorLayout {
        size_t num_blocks;
        size_t num_heads;
        size_t block_size;
        size_t key_head_size;
        size_t value_head_size;
        size_t query_head_size;  // may be zero if derivable from query tensor
    };

    inline OperatorLayout GetOperatorLayout() const {
        return cache_manager.get_operator_layout(handle);
    }
};

// ============================================================================
// Cache write helpers (copy one token across heads into its block/offset)
// ============================================================================
template <typename T>
inline void CopyTokenKeyValueIntoCache(const PagedAttentionKernelContext& ctx,
                                       size_t token_index,
                                       size_t sequence_index) {
    const T* key_src = static_cast<const T*>(ctx.key);
    const T* value_src = static_cast<const T*>(ctx.value);
    T* key_dst = static_cast<T*>(ctx.key_cache_base);
    T* value_dst = static_cast<T*>(ctx.value_cache_base);

    const size_t local_index_in_sequence = token_index - static_cast<size_t>(ctx.subsequence_begins[sequence_index]);
    const size_t offset_in_block = local_index_in_sequence % ctx.block_size;

    const int32_t block_id =
        AcquireOrRecycleBlockForSequence(sequence_index,
                                         ctx.block_indices,
                                         ctx.block_indices_begins,
                                         const_cast<std::vector<int32_t>&>(ctx.sequence_block_count),
                                         ctx.block_count);

    const int32_t begin = ctx.block_indices_begins[sequence_index];
    const int32_t tail = ctx.sequence_block_count[sequence_index] - 1;
    ctx.block_indices[begin + tail] = block_id;

    for (size_t h = 0; h < ctx.head_count; ++h) {
        const size_t key_cache_offset =
            ((static_cast<size_t>(block_id) * ctx.head_count + h) * ctx.block_size + offset_in_block) *
            ctx.key_head_size;
        const T* key_src_vec = key_src + token_index * ctx.key_feature_size + h * ctx.key_head_size;
        std::memcpy(key_dst + key_cache_offset, key_src_vec, ctx.key_head_size * sizeof(T));

        const size_t value_cache_offset =
            ((static_cast<size_t>(block_id) * ctx.head_count + h) * ctx.block_size + offset_in_block) *
            ctx.value_head_size;
        const T* value_src_vec = value_src + token_index * ctx.value_feature_size + h * ctx.value_head_size;
        std::memcpy(value_dst + value_cache_offset, value_src_vec, ctx.value_head_size * sizeof(T));
    }
}

inline bool TryResolveCachedBlockAndOffset(size_t sequence_index,
                                           int32_t key_position_in_context,
                                           const PagedAttentionKernelContext& ctx,
                                           int32_t& out_block_id,
                                           int32_t& out_offset_in_block) {
    const int32_t begin = ctx.block_indices_begins[sequence_index];
    const int32_t end = ctx.block_indices_begins[sequence_index + 1];
    const int32_t count = end - begin;

    if (key_position_in_context < count) {
        out_block_id = ctx.block_indices[begin + key_position_in_context];
        out_offset_in_block = key_position_in_context % static_cast<int32_t>(ctx.block_size);
        return true;
    }
    return false;
}

template <typename T>
inline T ComputeScoreAgainstCachedKey(const T* query_head_vector,
                                      size_t head_index,
                                      const PagedAttentionKernelContext& ctx,
                                      int32_t block_id,
                                      int32_t offset_in_block) {
    const T* key_cache = static_cast<const T*>(ctx.key_cache_base);
    const T* key_vec = key_cache + ((static_cast<size_t>(block_id) * ctx.head_count + head_index) * ctx.block_size +
                                    static_cast<size_t>(offset_in_block)) *
                                       ctx.key_head_size;

    T score = pa_math::ComputeDotProduct(query_head_vector, key_vec, ctx.key_head_size);

    int32_t rotated_index = -1;
    if (pa_rotary::BlockHasRotary(block_id, ctx.rotated_block_indices, ctx.rotated_block_count, rotated_index)) {
        std::vector<T> rotated(key_vec, key_vec + ctx.key_head_size);
        const int32_t trig_row = pa_rotary::ComputeTrigRowIndex(rotated_index,
                                                                offset_in_block,
                                                                ctx.rotation_deltas,
                                                                ctx.rotation_deltas_dim,
                                                                ctx.block_size);
        pa_rotary::ApplyRotaryEmbeddingToVector(rotated.data(),
                                                ctx.key_head_size,
                                                static_cast<const T*>(ctx.rotation_trig_lut),
                                                static_cast<size_t>(trig_row));
        score = pa_math::ComputeDotProduct(query_head_vector, rotated.data(), ctx.key_head_size);
    }

    return score;
}

template <typename T>
inline T ComputeScoreAgainstNewKey(const T* query_head_vector,
                                   size_t head_index,
                                   int32_t absolute_token_index,
                                   const PagedAttentionKernelContext& ctx) {
    const T* key_src = static_cast<const T*>(ctx.key);
    const T* key_vec =
        key_src + static_cast<size_t>(absolute_token_index) * ctx.key_feature_size + head_index * ctx.key_head_size;
    return pa_math::ComputeDotProduct(query_head_vector, key_vec, ctx.key_head_size);
}

template <typename T>
inline void AccumulateValueFromCachedKey(size_t head_index,
                                         const PagedAttentionKernelContext& ctx,
                                         int32_t block_id,
                                         int32_t offset_in_block,
                                         T weight,
                                         std::vector<T>& output_head_vector) {
    const T* value_cache = static_cast<const T*>(ctx.value_cache_base);
    const T* value_vec = value_cache + ((static_cast<size_t>(block_id) * ctx.head_count + head_index) * ctx.block_size +
                                        static_cast<size_t>(offset_in_block)) *
                                           ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i)
        output_head_vector[i] += weight * value_vec[i];
}

template <typename T>
inline void AccumulateValueFromNewKey(int32_t absolute_token_index,
                                      size_t head_index,
                                      const PagedAttentionKernelContext& ctx,
                                      T weight,
                                      std::vector<T>& output_head_vector) {
    const T* value_src = static_cast<const T*>(ctx.value);
    const T* value_vec = value_src + static_cast<size_t>(absolute_token_index) * ctx.value_feature_size +
                         head_index * ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i)
        output_head_vector[i] += weight * value_vec[i];
}

// ============================================================================
// Paged Attention kernel
// ============================================================================
template <typename T>
void paged_attention(T* out,         // [batch_tokens, head_count * value_head_size]
                     T* out_scores,  // [batch_tokens, head_count, max_context_length]

                     const T* query,  // [batch_tokens, head_count * query_head_size]
                     const T* key,    // [batch_tokens, head_count * key_head_size]
                     const T* value,  // [batch_tokens, head_count * value_head_size]

                     const int32_t* past_lens,               // [sequence_count]
                     const int32_t* subsequence_begins_opt,  // optional; if null, use CacheManager’s per-op view
                     int32_t* block_indices,                 // [block_count]
                     int32_t* block_indices_begins,          // [sequence_count + 1]

                     const T* scale_opt,                  // optional scalar
                     const int32_t* sliding_window_opt,   // optional scalar
                     const T* alibi_slopes_opt,           // [head_count] optional
                     const int32_t* max_context_len_opt,  // optional scalar

                     const int32_t* rotated_block_indices_opt,  // [rotated_block_count] optional
                     const int32_t* rotation_deltas_opt,        // [rotated_block_count x rotation_deltas_dim] optional
                     const T* rotation_trig_lut_opt,            // [rotation_lut_rows, key_head_size] optional

                     const ov::Shape& query_shape,
                     const ov::Shape& key_shape,
                     const ov::Shape& value_shape,
                     const ov::Shape& past_lens_shape,
                     const ov::Shape& rotated_block_indices_shape,
                     const ov::Shape& rotation_deltas_shape,
                     const ov::Shape& rotation_trig_lut_shape,

                     ov::internal::CacheManager::Handle cache_handle,
                     const std::shared_ptr<ov::internal::CacheManager>& cache_manager) {
    CacheManagerAdapter cm(*cache_manager, cache_handle);
    const auto layout = cm.GetOperatorLayout();

    PagedAttentionKernelContext ctx{};

    ctx.query = query;
    ctx.key = key;
    ctx.value = value;

    ctx.key_cache_base = cm.GetKeyCacheBase();
    ctx.value_cache_base = cm.GetValueCacheBase();

    ctx.past_lens = past_lens;

    const int32_t* subseq_from_cm = cm.GetSubsequenceBeginsOrNull();
    ctx.subsequence_begins = subsequence_begins_opt ? subsequence_begins_opt : subseq_from_cm;

    ctx.block_indices = block_indices;
    ctx.block_indices_begins = block_indices_begins;
    ctx.alibi_slopes = alibi_slopes_opt;

    ctx.rotated_block_indices = rotated_block_indices_opt;
    ctx.rotation_deltas = rotation_deltas_opt;
    ctx.rotation_trig_lut = rotation_trig_lut_opt;

    ctx.batch_token_count = static_cast<size_t>(query_shape[0]);
    ctx.query_feature_size = static_cast<size_t>(query_shape[1]);
    ctx.key_feature_size = static_cast<size_t>(key_shape[1]);
    ctx.value_feature_size = static_cast<size_t>(value_shape[1]);

    ctx.block_count = layout.num_blocks;
    ctx.head_count = layout.num_heads;
    ctx.block_size = layout.block_size;
    ctx.key_head_size = layout.key_head_size;
    ctx.value_head_size = layout.value_head_size;
    ctx.query_head_size = layout.query_head_size ? layout.query_head_size : (ctx.query_feature_size / ctx.head_count);

    ctx.sequence_count = static_cast<size_t>(past_lens_shape[0]);
    ctx.rotated_block_count =
        rotated_block_indices_shape.empty() ? 0 : static_cast<size_t>(rotated_block_indices_shape[0]);
    ctx.rotation_deltas_dim = (rotation_deltas_shape.empty() || ctx.rotated_block_count == 0)
                                  ? 0
                                  : static_cast<size_t>(rotation_deltas_shape[1]);
    ctx.rotation_lut_rows = rotation_trig_lut_shape.empty() ? 0 : static_cast<size_t>(rotation_trig_lut_shape[0]);

    ctx.max_context_length = max_context_len_opt ? max_context_len_opt[0] : 0;
    ctx.sliding_window = sliding_window_opt ? sliding_window_opt[0] : 0;

    ctx.sequence_block_count.resize(ctx.sequence_count);
    for (size_t s = 0; s < ctx.sequence_count; ++s) {
        const int32_t b = ctx.block_indices_begins[s];
        const int32_t e = ctx.block_indices_begins[s + 1];
        ctx.sequence_block_count[s] = e - b;
    }

    const T scale = scale_opt ? scale_opt[0] : T(1) / std::sqrt(static_cast<T>(ctx.key_head_size));

    pa_debug::PrintScalar(scale, "scale");

    for (size_t token_index = 0; token_index < ctx.batch_token_count; ++token_index) {
        const size_t sequence_index =
            ResolveSequenceIndexForToken(token_index, ctx.subsequence_begins, ctx.sequence_count);

        if (ctx.subsequence_begins && token_index >= static_cast<size_t>(ctx.subsequence_begins[sequence_index])) {
            CopyTokenKeyValueIntoCache<T>(ctx, token_index, sequence_index);
        }

        for (size_t head_index = 0; head_index < ctx.head_count; ++head_index) {
            const T* query_head_vector = static_cast<const T*>(ctx.query) + token_index * ctx.query_feature_size +
                                         head_index * ctx.query_head_size;

            const int32_t past_token_count = ctx.past_lens ? ctx.past_lens[sequence_index] : 0;

            const int32_t new_token_count =
                ctx.subsequence_begins
                    ? (ctx.subsequence_begins[sequence_index + 1] - ctx.subsequence_begins[sequence_index])
                    : 0;

            const int32_t total_key_count_unclamped = past_token_count + new_token_count;

            const int32_t total_key_count = (ctx.max_context_length > 0)
                                                ? std::min<int32_t>(total_key_count_unclamped, ctx.max_context_length)
                                                : total_key_count_unclamped;

            const int32_t keep_from =
                (ctx.sliding_window > 0) ? std::max<int32_t>(0, total_key_count - ctx.sliding_window) : 0;

            std::vector<T> head_scores(static_cast<size_t>(total_key_count), T(0));

            for (int32_t k = 0; k < total_key_count; ++k) {
                if (ctx.sliding_window > 0 && k < keep_from) {
                    head_scores[k] = -std::numeric_limits<T>::infinity();
                    continue;
                }

                T score_value = T(0);
                if (k < past_token_count) {
                    int32_t block_id, offset_in_block;
                    if (TryResolveCachedBlockAndOffset(sequence_index, k, ctx, block_id, offset_in_block)) {
                        score_value =
                            ComputeScoreAgainstCachedKey(query_head_vector, head_index, ctx, block_id, offset_in_block);
                    }
                } else {
                    const int32_t absolute_token_index =
                        ctx.subsequence_begins ? (ctx.subsequence_begins[sequence_index] + (k - past_token_count))
                                               : (k - past_token_count);
                    score_value = ComputeScoreAgainstNewKey(query_head_vector, head_index, absolute_token_index, ctx);
                }

                const T alibi = ctx.alibi_slopes ? static_cast<const T*>(ctx.alibi_slopes)[head_index] : T(0);
                head_scores[k] = score_value * scale + alibi * T(-(total_key_count - k - 1));
            }

            pa_math::ComputeSoftmaxInPlace(head_scores);

            std::vector<T> head_output(ctx.value_head_size, T(0));
            for (int32_t k = 0; k < total_key_count; ++k) {
                const T weight = head_scores[k];
                if (k < past_token_count) {
                    int32_t block_id, offset_in_block;
                    if (TryResolveCachedBlockAndOffset(sequence_index, k, ctx, block_id, offset_in_block)) {
                        AccumulateValueFromCachedKey(head_index, ctx, block_id, offset_in_block, weight, head_output);
                    }
                } else {
                    const int32_t absolute_token_index =
                        ctx.subsequence_begins ? (ctx.subsequence_begins[sequence_index] + (k - past_token_count))
                                               : (k - past_token_count);
                    AccumulateValueFromNewKey<T>(absolute_token_index, head_index, ctx, weight, head_output);
                }

                const size_t score_out_index =
                    (token_index * ctx.head_count + head_index) * static_cast<size_t>(ctx.max_context_length) +
                    static_cast<size_t>(k);
                out_scores[score_out_index] = head_scores[k];
            }

            T* out_head_dst = out + token_index * ctx.value_feature_size + head_index * ctx.value_head_size;
            std::memcpy(out_head_dst, head_output.data(), ctx.value_head_size * sizeof(T));
        }
    }
}

}  // namespace ov::reference
