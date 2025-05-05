// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/attr_types.hpp"
namespace ov::reference {
namespace paged_attention_utils {

//---------------------------------------------------------------------------
// Context: holds all parameters and computed dimensions in one place.
//---------------------------------------------------------------------------
struct PagedAttentionContext {
    // Outputs ptrs
    void* out;
    void* score;
    int32_t* updated_block_indices;
    int32_t* updated_block_indices_begins;
    int32_t* updated_free_block_indices;

    // Attention with cache inputs ptrs
    const void* query;
    const void* key;
    const void* value;
    void* key_cache;
    void* value_cache;

    // Sequencing related input ptrs
    const int32_t* past_lens;
    const int32_t* subsequence_begins;
    const int32_t* block_indices;
    const int32_t* block_indices_begins;
    const int32_t* max_blocks;
    const void* alibi_slopes;

    // Per-sequence block counter
    std::vector<int32_t> sequence_block_count;

    // Rotation parameters
    const int32_t* rotated_block_indices;
    const int32_t* rotation_deltas;
    const void* rotation_trig_lut;

    // Dimensions info
    size_t batch_tokens;
    size_t batch_sequence_count;
    size_t num_heads;
    size_t block_size;
    size_t num_blocks;
    size_t query_head_size;
    size_t key_head_size;
    size_t value_head_size;
    size_t query_feature_size;
    size_t key_feature_size;
    size_t value_feature_size;
    int32_t max_context_length;
    int32_t sliding_window;
    size_t num_rotated_blocks;
    size_t rotation_lut_rows;
    size_t rotation_deltas_dim;
};

//---------------------------------------------------------------------------
// Basic vector math
//---------------------------------------------------------------------------
template <typename T>
T dot_product(const T* a, const T* b, size_t length) {
    return std::inner_product(a, a + length, b, T(0));
}

template <typename T>
void softmax(std::vector<T>& values) {
    T max_value = *std::max_element(values.begin(), values.end()), sum = T(0);
    for (auto& v : values) {
        v = std::exp(v - max_value);
        sum += v;
    }
    for (auto& v : values)
        v /= sum;
}

//---------------------------------------------------------------------------
// Apply rotary positional embedding
//---------------------------------------------------------------------------
template <typename T>
void apply_rope(T* vector, size_t head_size, const T* trig_lut, size_t trig_index) {
    size_t half = head_size / 2;
    const T* row = trig_lut + trig_index * head_size;
    for (size_t i = 0; i < half; ++i) {
        T x0 = vector[2 * i], x1 = vector[2 * i + 1];
        vector[2 * i] = x0 * row[i] - x1 * row[half + i];
        vector[2 * i + 1] = x0 * row[half + i] + x1 * row[i];
    }
}

//---------------------------------------------------------------------------
// Rotation helpers
//---------------------------------------------------------------------------
inline bool should_apply_rotation(int32_t block_id, const PagedAttentionContext& ctx, int32_t& rotated_index) {
    if (!ctx.rotated_block_indices)
        return false;
    for (size_t i = 0; i < ctx.num_rotated_blocks; ++i) {
        if (ctx.rotated_block_indices[i] == block_id) {
            rotated_index = int32_t(i);
            return true;
        }
    }
    return false;
}

inline int32_t get_trig_index(int32_t rotated_index, int32_t token_offset, const PagedAttentionContext& ctx) {
    if (!ctx.rotation_deltas)
        return 0;
    if (ctx.rotation_deltas_dim == 1)
        return ctx.rotation_deltas[rotated_index];
    return ctx.rotation_deltas[rotated_index * ctx.block_size + token_offset];
}

//---------------------------------------------------------------------------
// Determine sequence index for a token
//---------------------------------------------------------------------------
inline size_t get_sequence_index(size_t token_index, const PagedAttentionContext& ctx) {
    if (ctx.batch_sequence_count <= 1 || !ctx.subsequence_begins)
        return 0;
    for (size_t s = 0; s < ctx.batch_sequence_count; ++s) {
        if (token_index >= size_t(ctx.subsequence_begins[s]) && token_index < size_t(ctx.subsequence_begins[s + 1]))
            return s;
    }
    return 0;
}

//---------------------------------------------------------------------------
// Allocate or evict a block for a sequence
//---------------------------------------------------------------------------
inline int32_t allocate_block_for_sequence(size_t sequence_index, PagedAttentionContext& ctx) {
    int32_t begin = ctx.updated_block_indices_begins[sequence_index];
    int32_t count = ctx.sequence_block_count[sequence_index];
    int32_t maxb = ctx.max_blocks[sequence_index];

    // Use free blokc from the list if below the limit
    if (count < maxb) {
        int32_t free_block = ctx.updated_free_block_indices[begin + count];
        ctx.sequence_block_count[sequence_index] = count + 1;
        return free_block;
    }
    // Otherwise evict oldest
    int32_t oldest = ctx.updated_block_indices[begin];
    for (int32_t i = 0; i + 1 < count; ++i)
        ctx.updated_block_indices[begin + i] = ctx.updated_block_indices[begin + i + 1];
    return oldest;
}

//---------------------------------------------------------------------------
// Insert new token into KV cache
//---------------------------------------------------------------------------
template <typename T>
void insert_new_token_into_cache(PagedAttentionContext& ctx, size_t token_index, size_t sequence_index) {
    const T* key_ptr = static_cast<const T*>(ctx.key);
    const T* value_ptr = static_cast<const T*>(ctx.value);
    T* key_cache_ptr = static_cast<T*>(ctx.key_cache);
    T* value_cache_ptr = static_cast<T*>(ctx.value_cache);

    size_t local_index = token_index - ctx.subsequence_begins[sequence_index];
    size_t offset = local_index % ctx.block_size;
    int32_t block_id = allocate_block_for_sequence(sequence_index, ctx);

    // Record new block in updated indices
    int32_t begin = ctx.updated_block_indices_begins[sequence_index];
    ctx.updated_block_indices[begin + ctx.sequence_block_count[sequence_index] - 1] = block_id;

    // Copy key and value vectors
    for (size_t head = 0; head < ctx.num_heads; ++head) {
        size_t k_offset = ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.key_head_size;
        std::memcpy(key_cache_ptr + k_offset,
                    key_ptr + token_index * ctx.key_feature_size + head * ctx.key_head_size,
                    ctx.key_head_size * sizeof(T));

        size_t v_offset = ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.value_head_size;
        std::memcpy(value_cache_ptr + v_offset,
                    value_ptr + token_index * ctx.value_feature_size + head * ctx.value_head_size,
                    ctx.value_head_size * sizeof(T));
    }
}

//---------------------------------------------------------------------------
// Find cached token for scoring/accumulation
//---------------------------------------------------------------------------
inline bool find_cached_token(size_t sequence_index,
                              int32_t token_position,
                              const PagedAttentionContext& ctx,
                              int32_t& block_id,
                              int32_t& offset) {
    int32_t begin = ctx.updated_block_indices_begins[sequence_index];
    int32_t end = ctx.updated_block_indices_begins[sequence_index + 1];
    int32_t count = end - begin;
    if (token_position < count) {
        block_id = ctx.updated_block_indices[begin + token_position];
        offset = token_position % ctx.block_size;
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------
// Compute score for cached key
//---------------------------------------------------------------------------
template <typename T>
T compute_score_for_cached_key(const T* query_vec,
                               size_t head,
                               const PagedAttentionContext& ctx,
                               int32_t block_id,
                               int32_t offset) {
    const T* cache_base = static_cast<const T*>(ctx.key_cache);
    const T* key_vec = cache_base + ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.key_head_size;

    T score = dot_product(query_vec, key_vec, ctx.key_head_size);
    int32_t rot_idx;
    if (should_apply_rotation(block_id, ctx, rot_idx)) {
        std::vector<T> tmp(key_vec, key_vec + ctx.key_head_size);
        apply_rope(tmp.data(),
                   ctx.key_head_size,
                   static_cast<const T*>(ctx.rotation_trig_lut),
                   get_trig_index(rot_idx, offset, ctx));
        score = dot_product(query_vec, tmp.data(), ctx.key_head_size);
    }
    return score;
}

//---------------------------------------------------------------------------
// Compute score for new key
//---------------------------------------------------------------------------
template <typename T>
T compute_score_for_new_key(const T* query_vec, size_t head, int32_t new_token_idx, const PagedAttentionContext& ctx) {
    const T* key_ptr = static_cast<const T*>(ctx.key);
    const T* key_vec = key_ptr + size_t(new_token_idx) * ctx.key_feature_size + head * ctx.key_head_size;
    return dot_product(query_vec, key_vec, ctx.key_head_size);
}

//---------------------------------------------------------------------------
// Accumulate value from cached key
//---------------------------------------------------------------------------
template <typename T>
void accumulate_value_for_cached_key(size_t head,
                                     const PagedAttentionContext& ctx,
                                     int32_t block_id,
                                     int32_t offset,
                                     T weight,
                                     std::vector<T>& output) {
    const T* cache_base = static_cast<const T*>(ctx.value_cache);
    const T* val_vec = cache_base + ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i) {
        output[i] += weight * val_vec[i];
    }
}

//---------------------------------------------------------------------------
// Accumulate value from new key
//---------------------------------------------------------------------------
template <typename T>
void accumulate_value_for_new_key(int32_t new_token_idx,
                                  size_t head,
                                  const PagedAttentionContext& ctx,
                                  T weight,
                                  std::vector<T>& output) {
    const T* value_ptr = static_cast<const T*>(ctx.value);
    const T* val_vec = value_ptr + size_t(new_token_idx) * ctx.value_feature_size + head * ctx.value_head_size;
    for (size_t i = 0; i < ctx.value_head_size; ++i) {
        output[i] += weight * val_vec[i];
    }
}
}  // namespace paged_attention_utils

//---------------------------------------------------------------------------
// Main PagedAttention function
//---------------------------------------------------------------------------

template <typename T>
void paged_attention(T* out,                                 // Output attention result
                     T* score,                               // Output concatenated raw scores
                     int32_t* updated_block_indices,         // [num_blocks]
                     int32_t* updated_block_indices_begins,  // [batch_seq+1]
                     int32_t* updated_free_block_indices,    // [num_blocks]

                     const T* query,                       // [batch_tokens, num_heads * q_head_size]
                     const T* key,                         // [batch_tokens, num_heads * k_head_size]
                     const T* value,                       // [batch_tokens, num_heads * v_head_size]
                     T* key_cache,                         // [num_blocks, num_heads, block_size, k_head_size]
                     T* value_cache,                       // [num_blocks, num_heads, block_size, v_head_size]
                     const int32_t* past_lens,             // [batch_seq]
                     const int32_t* subsequence_begins,    // [batch_seq+1]
                     const int32_t* block_indices,         // [num_blocks]
                     const int32_t* block_indices_begins,  // [batch_seq+1]
                     const T* scale_ptr,                   // (Optional) scale factor
                     const int32_t* sliding_window_ptr,    // (Optional)
                     const T* alibi_slopes,                // (Optional) [num_heads]
                     const int32_t* max_context_len_ptr,   // (Optional)

                     const int32_t* rotated_block_indices,  // (Optional) [num_rotated_blocks]
                     const int32_t* rotation_deltas,        // (Optional) [num_rotated_blocks × rotation_deltas_dim]
                     const T* rotation_trig_lut,            // (Optional) [rotation_lut_rows, head_size]
                     const int32_t* free_block_indices,     // [num_blocks]
                     const int32_t* max_blocks,             // [batch_seq]

                     const ov::Shape& query_shape,
                     const ov::Shape& key_shape,
                     const ov::Shape& value_shape,
                     const ov::Shape& key_cache_shape,
                     const ov::Shape& value_cache_shape,
                     const ov::Shape& past_lens_shape,
                     const ov::Shape& rotated_block_indices_shape,
                     const ov::Shape& rotation_deltas_shape,
                     const ov::Shape& rotation_trig_lut_shape) {
    using namespace paged_attention_utils;

    // Build context
    PagedAttentionContext ctx{};
    ctx.out = out;
    ctx.score = score;
    ctx.query = query;
    ctx.key = key;
    ctx.value = value;
    ctx.key_cache = key_cache;
    ctx.value_cache = value_cache;
    ctx.alibi_slopes = alibi_slopes;
    ctx.rotation_trig_lut = rotation_trig_lut;
    ctx.past_lens = past_lens;
    ctx.subsequence_begins = subsequence_begins;
    ctx.block_indices = block_indices;
    ctx.block_indices_begins = block_indices_begins;
    ctx.rotated_block_indices = rotated_block_indices;
    ctx.rotation_deltas = rotation_deltas;
    ctx.updated_block_indices = updated_block_indices;
    ctx.updated_block_indices_begins = updated_block_indices_begins;
    ctx.updated_free_block_indices = updated_free_block_indices;
    ctx.max_blocks = max_blocks;

    // Populate dimensions
    ctx.batch_tokens = query_shape[0];
    ctx.query_feature_size = query_shape[1];
    ctx.key_feature_size = key_shape[1];
    ctx.value_feature_size = value_shape[1];

    ctx.num_blocks = key_cache_shape[0];
    ctx.num_heads = key_cache_shape[1];
    ctx.block_size = key_cache_shape[2];
    ctx.key_head_size = key_cache_shape[3];
    ctx.value_head_size = value_cache_shape[3];

    ctx.query_head_size = ctx.query_feature_size / ctx.num_heads;

    ctx.batch_sequence_count = past_lens_shape[0];

    ctx.num_rotated_blocks = rotated_block_indices_shape.empty() ? 0 : rotated_block_indices_shape[0];
    ctx.rotation_deltas_dim =
        rotation_deltas_shape.empty() || ctx.num_rotated_blocks == 0 ? 0 : rotation_deltas_shape[1];
    ctx.rotation_lut_rows = rotation_trig_lut_shape.empty() ? 0 : rotation_trig_lut_shape[0];
    ctx.max_context_length = max_context_len_ptr ? max_context_len_ptr[0] : 0;
    ctx.sliding_window = sliding_window_ptr ? sliding_window_ptr[0] : 0;

    // Initialize updated arrays
    std::memcpy(updated_block_indices, block_indices, num_blocks * sizeof(int32_t));
    std::memcpy(updated_block_indices_begins, block_indices_begins, (batch_sequence_count + 1) * sizeof(int32_t));
    std::memcpy(updated_free_block_indices, free_block_indices, num_blocks * sizeof(int32_t));

    // Seed per-sequence counts
    ctx.sequence_block_count.resize(batch_sequence_count);
    for (size_t s = 0; s < batch_sequence_count; ++s) {
        int32_t begin = updated_block_indices_begins[s];
        int32_t end = updated_block_indices_begins[s + 1];
        ctx.sequence_block_count[s] = end - begin;
    }

    // Compute scale factor
    T scale = scale_ptr ? scale_ptr[0] : T(1) / std::sqrt(T(value_head_size));

    // Loop over tokens
    for (size_t token_idx = 0; token_idx < batch_tokens; ++token_idx) {
        size_t seq_idx = get_sequence_index(token_idx, ctx);

        // Update KV cache if new token in this sequence
        if (subsequence_begins && token_idx >= size_t(subsequence_begins[seq_idx])) {
            insert_new_token_into_cache<T>(ctx, token_idx, seq_idx);
        }

        // Per-head attention
        for (size_t head = 0; head < num_heads; ++head) {
            const T* q_vector = query + token_idx * query_feature_size + head * query_head_size;

            int32_t past_tokens = past_lens ? past_lens[seq_idx] : 0;
            int32_t new_tokens =
                subsequence_begins ? (subsequence_begins[seq_idx + 1] - subsequence_begins[seq_idx]) : 0;
            int32_t total_keys = std::min(past_tokens + new_tokens, ctx.max_context_length);

            std::vector<T> scores(total_keys, T(0));

            // Compute raw attention scores
            for (int32_t k = 0; k < total_keys; ++k) {
                T score_value = T(0);
                if (k < past_tokens) {
                    int32_t block_id, offset;
                    if (find_cached_token(seq_idx, k, ctx, block_id, offset)) {
                        if (block_id == ctx.block_indices[ctx.block_indices_begins[seq_idx]] &&
                            offset < ctx.sliding_window) {
                            scores[k] = -std::numeric_limits<T>::infinity();
                            continue;
                        } else {
                            score_value = compute_score_for_cached_key(q_vector, head, ctx, block_id, offset);
                        }
                    }
                } else {
                    int32_t new_idx =
                        subsequence_begins ? (subsequence_begins[seq_idx] + (k - past_tokens)) : (k - past_tokens);
                    score_value = compute_score_for_new_key(q_vector, head, new_idx, ctx);
                }
                T alibi = alibi_slopes ? ((T*)alibi_slopes)[head] : T(0);
                scores[k] = score_value * scale + alibi * T(-(total_keys - k - 1));
            }

            softmax(scores);

            // Accumulate values
            std::vector<T> output_vector(value_head_size, T(0));
            for (int32_t k = 0; k < total_keys; ++k) {
                T weight = scores[k];
                if (k < past_tokens) {
                    int32_t block_id, offset;
                    if (find_cached_token(seq_idx, k, ctx, block_id, offset)) {
                        if (!(block_id == ctx.block_indices[ctx.block_indices_begins[seq_idx]] &&
                              offset < ctx.sliding_window)) {
                            accumulate_value_for_cached_key(head, ctx, block_id, offset, weight, output_vector);
                        } else {
                            continue;
                        }
                    }
                } else {
                    int32_t new_idx =
                        subsequence_begins ? (subsequence_begins[seq_idx] + (k - past_tokens)) : (k - past_tokens);
                    accumulate_value_for_new_key<T>(new_idx, head, ctx, weight, output_vector);
                }
                size_t global_index = seq_idx * ctx.max_context_length + k;
                score[global_index] = scores[k];
            }

            // Copy output
            T* destination = out + token_idx * value_feature_size + head * value_head_size;
            std::memcpy(destination, output_vector.data(), value_head_size * sizeof(T));
        }
    }
}

}  // namespace ov::reference