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
// Context Structure: Bundles common parameters (grouped by use and type)
//---------------------------------------------------------------------------

// Data pointers (all pointers are cast as needed)
struct PagedAttentionContext {
    // Output and primary input buffers
    void* out;                      // Output attention result
    void* score;                    // Output concatenated raw scores
    const void* query;              // [batch_tokens, num_q_heads * head_size]
    const void* key;                // [batch_tokens, num_k_heads * head_size]
    const void* value;              // [batch_tokens, num_v_heads * head_size]
    void* key_cache;                // [num_blocks, num_k_heads, block_size, head_size]
    void* value_cache;              // [num_blocks, num_v_heads, block_size, head_size]
    const void* alibi_slopes;       // (Optional) [num_k_heads]: per-head bias slopes
    const void* rotation_trig_lut;  // (Optional) [lut_rows, head_size]

    // Shape information
    ov::Shape q_shape;                      // [batch_tokens, num_q_heads * head_size]
    ov::Shape key_shape;                    // [batch_tokens, num_k_heads * head_size]
    ov::Shape value_shape;                  // [batch_tokens, num_v_heads * head_size]
    ov::Shape key_cache_shape;              // [num_blocks, num_k_heads, block_size, head_size]
    ov::Shape value_cache_shape;            // [num_blocks, num_v_heads, block_size, head_size]
    ov::Shape past_lens_shape;              // [batch_seq]
    ov::Shape rotated_block_indices_shape;  // (Optional) [num_rotated_blocks]
    ov::Shape rotation_deltas_shape;        // (Optional) [num_rotated_blocks, block_size or 1]
    ov::Shape rotation_trig_lut_shape;      // (Optional) [lut_rows, head_size]

    // Auxiliary pointers
    const int32_t* past_lens;              // [batch_seq]: number of past tokens per sequence
    const int32_t* subsequence_begins;     // [batch_seq + 1]: start indices of new tokens per sequence
    const int32_t* block_indices;          // [num_blocks]: block table for each sequence
    const int32_t* block_indices_begins;   // [batch_seq + 1]: indices into block_indices per sequence
    const int32_t* sliding_window_ptr;     // (Optional) sliding window parameter
    const int32_t* max_context_len_ptr;    // (Optional) maximum context length for score output indexing
    const int32_t* rotated_block_indices;  // (Optional) [num_rotated_blocks]
    const int32_t* rotation_deltas;        // (Optional) [num_rotated_blocks, block_size or 1]

    // Computed dimensions (using size_t)
    size_t batch_tokens;    // number of tokens in the batch
    size_t batch_seq;       // number of sequences ([batch_seq])
    size_t query_features;  // equals num_q_heads * head_size

    // Head and dimension info
    size_t num_q_heads;  // computed from q_shape[1] / head_size
    size_t num_k_heads;  // from [num_blocks, num_k_heads, block_size, head_size]
    size_t num_v_heads;  // from [num_blocks, num_v_heads, block_size, head_size]
    size_t head_size;    // from [num_blocks, num_k_heads, block_size, head_size]
    size_t num_blocks;   // from [num_blocks, num_k_heads, block_size, head_size]
    size_t block_size;   // from [num_blocks, num_k_heads, block_size, head_size]

    // Optional parameters
    int32_t sliding_window;   // from sliding_window_ptr (if present)
    int32_t max_context_len;  // from max_context_len_ptr (if present)
};

//---------------------------------------------------------------------------
// Helper Functions
//---------------------------------------------------------------------------

template <typename T>
T dot_product(const T* a, const T* b, size_t size) {
    return std::inner_product(a, a + size, b, T(0));
}

template <typename T>
void softmax(std::vector<T>& scores) {
    T max_score = *std::max_element(scores.begin(), scores.end());
    T sum = T(0);
    for (auto& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }
    for (auto& s : scores)
        s /= sum;
}

template <typename T>
void apply_rope(T* vec, size_t head_size, const T* rotation_trig_lut, size_t trig_index) {
    size_t half = head_size / 2;
    const T* row = rotation_trig_lut + trig_index * head_size;
    for (size_t i = 0; i < half; i++) {
        T x0 = vec[2 * i];
        T x1 = vec[2 * i + 1];
        T cos_val = row[i];
        T sin_val = row[half + i];
        vec[2 * i] = x0 * cos_val - x1 * sin_val;
        vec[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

inline bool should_apply_rotation(int32_t block_id, const PagedAttentionContext& ctx, int32_t& rotated_index) {
    const int32_t* rbi = ctx.rotated_block_indices;
    if (!rbi || ctx.rotated_block_indices_shape.empty())
        return false;
    size_t num_rotated_blocks = ctx.rotated_block_indices_shape[0];
    for (size_t i = 0; i < num_rotated_blocks; i++) {
        if (rbi[i] == block_id) {
            rotated_index = static_cast<int32_t>(i);
            return true;
        }
    }
    return false;
}

inline int32_t get_trig_index(int32_t rotated_index, int32_t token_offset, const PagedAttentionContext& ctx) {
    int32_t trig_index = 0;
    const int32_t* rdelta = ctx.rotation_deltas;
    if (rdelta && !ctx.rotation_deltas_shape.empty() && ctx.rotation_deltas_shape.size() >= 2) {
        if (ctx.rotation_deltas_shape[1] == 1ul)
            trig_index = rdelta[rotated_index];
        else if (ctx.rotation_deltas_shape[1] == ctx.block_size)
            trig_index = rdelta[rotated_index * ctx.block_size + token_offset];
    }
    return trig_index;
}

inline void get_cache_block_id_and_offset(size_t new_token_index,
                                          const PagedAttentionContext& ctx,
                                          int32_t& block_id,
                                          int32_t& token_offset) {
    size_t block_position = new_token_index / ctx.block_size;
    token_offset = static_cast<int32_t>(new_token_index % ctx.block_size);
    block_id = static_cast<int32_t>(block_position % ctx.num_blocks);  // circular wrap-around
}

inline size_t get_sequence_index(size_t token_idx, const PagedAttentionContext& ctx) {
    size_t seq_idx = 0;
    if (ctx.batch_seq > 1 && ctx.subsequence_begins) {
        for (size_t s = 0; s < ctx.batch_seq; s++) {
            if (token_idx >= static_cast<size_t>(ctx.subsequence_begins[s]) &&
                token_idx < static_cast<size_t>(ctx.subsequence_begins[s + 1])) {
                seq_idx = s;
                break;
            }
        }
    }
    return seq_idx;
}

template <typename T>
void insert_new_token_into_cache(const PagedAttentionContext& ctx, size_t token_idx, size_t seq_idx) {
    const T* key = static_cast<const T*>(ctx.key);
    const T* value = static_cast<const T*>(ctx.value);
    T* key_cache = static_cast<T*>(ctx.key_cache);
    T* value_cache = static_cast<T*>(ctx.value_cache);
    const ov::Shape& key_shape = ctx.key_shape;
    const ov::Shape& value_shape = ctx.value_shape;

    size_t new_token_index = token_idx - ctx.subsequence_begins[seq_idx];
    int32_t block_id_new, token_offset;
    get_cache_block_id_and_offset(new_token_index, ctx, block_id_new, token_offset);
    for (size_t kh = 0; kh < ctx.num_k_heads; kh++) {
        size_t cache_idx = ((block_id_new * ctx.num_k_heads + kh) * ctx.block_size + token_offset) * ctx.head_size;
        const T* src_key = key + token_idx * key_shape[1] + kh * ctx.head_size;
        std::memcpy(key_cache + cache_idx, src_key, ctx.head_size * sizeof(T));
    }
    for (size_t vh = 0; vh < ctx.num_v_heads; vh++) {
        size_t cache_idx = ((block_id_new * ctx.num_v_heads + vh) * ctx.block_size + token_offset) * ctx.head_size;
        const T* src_val = value + token_idx * value_shape[1] + vh * ctx.head_size;
        std::memcpy(value_cache + cache_idx, src_val, ctx.head_size * sizeof(T));
    }
}

inline bool find_cached_token(size_t seq_idx,
                              size_t token_position,
                              const PagedAttentionContext& ctx,
                              int32_t& block_id,
                              size_t& token_offset) {
    if (!ctx.block_indices || !ctx.block_indices_begins || ctx.past_lens_shape.empty())
        return false;
    size_t start = ctx.block_indices_begins[seq_idx];
    size_t end = ctx.block_indices_begins[seq_idx + 1];
    // Use for loop instead?
    if (token_position < (end - start)) {
        block_id = ctx.block_indices[start + token_position];
        token_offset = token_position % ctx.block_size;
        return true;
    }
    return false;
}

template <typename T>
T compute_score_for_cached_key(const T* q_vec,
                               size_t h,
                               const PagedAttentionContext& ctx,
                               int32_t block_id,
                               int32_t token_offset) {
    size_t k_head = h / (ctx.num_q_heads / ctx.num_k_heads);
    T* key_cache = static_cast<T*>(ctx.key_cache);
    size_t cache_idx = ((block_id * ctx.num_k_heads + k_head) * ctx.block_size + token_offset) * ctx.head_size;
    const T* key_vec = key_cache + cache_idx;
    T score_val = T(0);
    int32_t rotated_index = -1;
    if (should_apply_rotation(block_id, ctx, rotated_index)) {
        int32_t trig_index = get_trig_index(rotated_index, token_offset, ctx);
        std::vector<T> temp_key(key_vec, key_vec + ctx.head_size);
        const T* rotation_trig_lut = static_cast<const T*>(ctx.rotation_trig_lut);
        apply_rope(temp_key.data(), ctx.head_size, rotation_trig_lut, trig_index);
        score_val = dot_product(q_vec, temp_key.data(), ctx.head_size);
    } else {
        score_val = dot_product(q_vec, key_vec, ctx.head_size);
    }
    return score_val;
}

template <typename T>
T compute_score_for_new_key(const T* q_vec, size_t h, size_t new_token_idx, const PagedAttentionContext& ctx) {
    const T* key = static_cast<const T*>(ctx.key);
    const ov::Shape& key_shape = ctx.key_shape;
    size_t k_head = h % ctx.num_k_heads;
    const T* key_vec = key + new_token_idx * key_shape[1] + k_head * ctx.head_size;
    return dot_product(q_vec, key_vec, ctx.head_size);
}

template <typename T>
void accumulate_value_for_cached_key(const T* q_vec,
                                     size_t h,
                                     const PagedAttentionContext& ctx,
                                     int32_t block_id,
                                     int32_t token_offset,
                                     T weight,
                                     std::vector<T>& out_vec) {
    T* value_cache = static_cast<T*>(ctx.value_cache);
    size_t v_head = h % ctx.num_v_heads;
    size_t cache_idx = ((block_id * ctx.num_v_heads + v_head) * ctx.block_size + token_offset) * ctx.head_size;
    const T* raw_val_vec = value_cache + cache_idx;
    int32_t rotated_index = -1;
    if (should_apply_rotation(block_id, ctx, rotated_index)) {
        int32_t trig_index = get_trig_index(rotated_index, token_offset, ctx);
        std::vector<T> temp_value(raw_val_vec, raw_val_vec + ctx.head_size);
        const T* rotation_trig_lut = static_cast<const T*>(ctx.rotation_trig_lut);
        apply_rope(temp_value.data(), ctx.head_size, rotation_trig_lut, trig_index);
        for (size_t d = 0; d < ctx.head_size; d++) {
            out_vec[d] += weight * temp_value[d];
        }
    } else {
        for (size_t d = 0; d < ctx.head_size; d++) {
            out_vec[d] += weight * raw_val_vec[d];
        }
    }
}

template <typename T>
void accumulate_value_for_new_key(size_t new_token_idx,
                                  size_t h,
                                  const PagedAttentionContext& ctx,
                                  T weight,
                                  std::vector<T>& out_vec) {
    const T* value = static_cast<const T*>(ctx.value);
    const ov::Shape& value_shape = ctx.value_shape;
    size_t v_head = h % ctx.num_v_heads;
    const T* val_vec = value + new_token_idx * value_shape[1] + v_head * ctx.head_size;
    for (size_t d = 0; d < ctx.head_size; d++) {
        out_vec[d] += weight * val_vec[d];
    }
}
}  // namespace paged_attention_utils


//---------------------------------------------------------------------------
// Main PagedAttention Function
//---------------------------------------------------------------------------
template <typename T>
void paged_attention(T* out,                               // Output attention result
                     T* score,                             // Output concatenated raw scores
                     const T* query,                       // [batch_tokens, num_q_heads * head_size]
                     const T* key,                         // [batch_tokens, num_k_heads * head_size]
                     const T* value,                       // [batch_tokens, num_v_heads * head_size]
                     T* key_cache,                         // [num_blocks, num_k_heads, block_size, head_size]
                     T* value_cache,                       // [num_blocks, num_v_heads, block_size, head_size]
                     const ov::Shape& q_shape,             // [batch_tokens, num_q_heads * head_size]
                     const ov::Shape& key_shape,           // [batch_tokens, num_k_heads * head_size]
                     const ov::Shape& value_shape,         // [batch_tokens, num_v_heads * head_size]
                     const ov::Shape& key_cache_shape,     // [num_blocks, num_k_heads, block_size, head_size]
                     const ov::Shape& value_cache_shape,   // [num_blocks, num_v_heads, block_size, head_size]
                     const int32_t* past_lens,             // [batch_seq]: number of past tokens per sequence
                     const ov::Shape& past_lens_shape,     // [batch_seq]
                     const int32_t* subsequence_begins,    // [batch_seq + 1]: start indices of new tokens per sequence
                     const int32_t* block_indices,         // [num_blocks]: block table for each sequence
                     const int32_t* block_indices_begins,  // [batch_seq + 1]: indices into block_indices per sequence
                     const T* scale_ptr,                   // (Optional) pointer to attention scale factor
                     const int32_t* sliding_window_ptr,    // (Optional) sliding window parameter
                     const T* alibi_slopes,                // (Optional) [num_k_heads]: per-head bias slopes
                     const int32_t* max_context_len_ptr,  // (Optional) maximum context length for score output indexing
                     // Rotation parameters (if any is nullptr, rotation is skipped)
                     const int32_t* rotated_block_indices,          // (Optional) [num_rotated_blocks]
                     const int32_t* rotation_deltas,                // (Optional) [num_rotated_blocks, block_size or 1]
                     const T* rotation_trig_lut,                    // (Optional) [lut_rows, head_size]
                     const ov::Shape& rotated_block_indices_shape,  // (Optional) [num_rotated_blocks]
                     const ov::Shape& rotation_deltas_shape,        // (Optional) [num_rotated_blocks, block_size or 1]
                     const ov::Shape& rotation_trig_lut_shape       // (Optional) [lut_rows, head_size]
) {
    //-------------------------------------------------------------------------
    // Build context structure from input parameters.
    //-------------------------------------------------------------------------
    paged_attention_utils::PagedAttentionContext ctx;

    // Data pointers
    ctx.out = out;
    ctx.score = score;
    ctx.query = query;
    ctx.key = key;
    ctx.value = value;
    ctx.key_cache = key_cache;
    ctx.value_cache = value_cache;
    ctx.alibi_slopes = alibi_slopes;
    ctx.rotation_trig_lut = rotation_trig_lut;

    // Shape information
    ctx.q_shape = q_shape;
    ctx.key_shape = key_shape;
    ctx.value_shape = value_shape;
    ctx.key_cache_shape = key_cache_shape;
    ctx.value_cache_shape = value_cache_shape;
    ctx.past_lens_shape = past_lens_shape;
    ctx.rotated_block_indices_shape = rotated_block_indices_shape;
    ctx.rotation_deltas_shape = rotation_deltas_shape;
    ctx.rotation_trig_lut_shape = rotation_trig_lut_shape;

    // Auxiliary pointers
    ctx.past_lens = past_lens;
    ctx.subsequence_begins = subsequence_begins;
    ctx.block_indices = block_indices;
    ctx.block_indices_begins = block_indices_begins;
    ctx.sliding_window_ptr = sliding_window_ptr;
    ctx.max_context_len_ptr = max_context_len_ptr;
    ctx.rotated_block_indices = rotated_block_indices;
    ctx.rotation_deltas = rotation_deltas;

    // Computed dimensions (using size_t)
    ctx.batch_tokens = q_shape[0];
    ctx.query_features = q_shape[1];  // equals num_q_heads * head_size
    ctx.head_size = key_cache_shape[3];
    ctx.num_q_heads = ctx.head_size ? ctx.query_features / ctx.head_size : 0;
    ctx.num_blocks = key_cache_shape[0];
    ctx.num_k_heads = key_cache_shape[1];
    ctx.block_size = key_cache_shape[2];
    ctx.num_v_heads = value_cache_shape[1];
    ctx.batch_seq = !past_lens_shape.empty() ? past_lens_shape[0] : 1;
    ctx.sliding_window = sliding_window_ptr ? sliding_window_ptr[0] : 0;
    ctx.max_context_len = max_context_len_ptr ? max_context_len_ptr[0] : 0;

    // Compute scale (using provided scale_ptr; if null, default is 1/sqrt(head_size))
    T scale = scale_ptr ? scale_ptr[0] : T(1) / std::sqrt(static_cast<T>(ctx.head_size));

    //-------------------------------------------------------------------------
    // Process each query token using the constructed context.
    //-------------------------------------------------------------------------
    for (size_t token_idx = 0; token_idx < ctx.batch_tokens; token_idx++) {
        size_t seq_idx = paged_attention_utils::get_sequence_index(token_idx, ctx);

        // Insert new token into cache if needed.
        if (ctx.subsequence_begins && token_idx >= static_cast<size_t>(ctx.subsequence_begins[seq_idx])) {
            paged_attention_utils::insert_new_token_into_cache<T>(ctx, token_idx, seq_idx);
        }

        // Process each query head.
        for (size_t h = 0; h < ctx.num_q_heads; h++) {
            const T* q_vec = query + token_idx * ctx.query_features + h * ctx.head_size;
            size_t seq_new_tokens =
                ctx.subsequence_begins ? (ctx.subsequence_begins[seq_idx + 1] - ctx.subsequence_begins[seq_idx]) : 0;
            size_t seq_past_tokens = past_lens ? static_cast<size_t>(past_lens[seq_idx]) : 0;
            size_t total_keys = seq_past_tokens + seq_new_tokens;

            std::vector<T> scores(total_keys, T(0));
            // Compute attention scores.
            for (size_t k = 0; k < total_keys; k++) {
                T score_val = T(0);
                if (k < seq_past_tokens) {
                    int32_t block_id = -1;
                    size_t token_offset = 0;
                    if (!paged_attention_utils::find_cached_token(seq_idx, k, ctx, block_id, token_offset))
                        continue;
                    int32_t first_block_for_seq =
                        (ctx.block_indices_begins ? ctx.block_indices[ctx.block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq &&
                        token_offset < ctx.sliding_window)
                        score_val = -std::numeric_limits<T>::infinity();
                    else
                        score_val = paged_attention_utils::compute_score_for_cached_key(q_vec, h, ctx, block_id, token_offset);
                } else {
                    size_t new_token_idx = ctx.subsequence_begins
                                               ? (ctx.subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                               : (k - seq_past_tokens);
                    score_val = paged_attention_utils::compute_score_for_new_key(q_vec, h, new_token_idx, ctx);
                }
                T alibi = alibi_slopes ? alibi_slopes[h % ctx.num_k_heads] : T(0);
                score_val = score_val * scale + alibi * static_cast<T>(-(total_keys - k - 1));
                scores[k] = score_val;
            }
            paged_attention_utils::softmax(scores);

            // Compute weighted sum over value vectors.
            std::vector<T> out_vec(ctx.head_size, T(0));
            for (size_t k = 0; k < total_keys; k++) {
                T weight = scores[k];
                if (k < seq_past_tokens) {
                    int32_t block_id = -1;
                    int32_t token_offset = 0;
                    if (!paged_attention_utils::find_cached_token(seq_idx, k, ctx, block_id, token_offset))
                        continue;
                    int32_t first_block_for_seq =
                        (ctx.block_indices_begins ? ctx.block_indices[ctx.block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq &&
                        token_offset < ctx.sliding_window)
                        continue;
                        paged_attention_utils::accumulate_value_for_cached_key(q_vec, h, ctx, block_id, token_offset, weight, out_vec);
                } else {
                    size_t new_token_idx = ctx.subsequence_begins
                                               ? (ctx.subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                               : (k - seq_past_tokens);
                                               paged_attention_utils::accumulate_value_for_new_key<T>(new_token_idx, h, ctx, weight, out_vec);
                }
                size_t global_score_index = seq_idx * static_cast<size_t>(ctx.max_context_len) + k;
                score[global_score_index] = scores[k];
            }
            T* dst = out + token_idx * ctx.query_features + h * ctx.head_size;
            std::memcpy(dst, out_vec.data(), ctx.head_size * sizeof(T));
        }
    }
}

}  // namespace ov::reference
