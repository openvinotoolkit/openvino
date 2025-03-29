// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov::reference {
namespace paged_attention_utils {

// --- Helper / Unit Functions ---

template <typename T>
T dot_product(const T* a, const T* b, int32_t size) {
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

/*
 * Apply RoPE (Rotary Positional Embedding) rotation to a vector.
 */
template <typename T>
void apply_rope(T* vec, int32_t head_size, const T* rotation_trig_lut, int32_t trig_index) {
    int32_t half = head_size / 2;
    const T* row = rotation_trig_lut + trig_index * head_size;
    for (int32_t i = 0; i < half; i++) {
        T x0 = vec[2 * i];
        T x1 = vec[2 * i + 1];
        T cos_val = row[i];
        T sin_val = row[half + i];
        vec[2 * i] = x0 * cos_val - x1 * sin_val;
        vec[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/*
 * Look up the cached block token information.
 * Returns true if a valid cached token was found.
 */
bool find_cached_token(int32_t seq_idx,
                       int32_t token_idx,  // token index in cached keys (relative: 0..seq_past_tokens-1)
                       const int32_t* block_indices,
                       const int32_t* block_indices_begins,
                       int32_t num_blocks,
                       int32_t block_size,
                       int32_t& block_id,
                       int32_t& token_offset) {
    int32_t block_start = block_indices_begins ? block_indices_begins[seq_idx] : 0;
    int32_t block_end = block_indices_begins ? block_indices_begins[seq_idx + 1] : num_blocks;
    int32_t remaining = token_idx;
    for (int32_t b = block_start; b < block_end; b++) {
        if (remaining < block_size) {
            block_id = block_indices[b];
            token_offset = remaining;
            return true;
        }
        remaining -= block_size;
    }
    return false;
}

/*
 * Given rotation_deltas and its shape, compute the trig index for RoPE.
 * Returns 0 if rotation parameters are not available.
 */
int32_t get_trig_index(const int32_t* rotation_deltas,
                       const ov::Shape& rotation_deltas_shape,
                       int32_t rotated_index,
                       int32_t token_offset,
                       int32_t block_size) {
    int32_t trig_index = 0;
    if (rotation_deltas && !rotation_deltas_shape.empty() && rotation_deltas_shape.size() >= 2) {
        if (rotation_deltas_shape[1] == 1ul)
            trig_index = rotation_deltas[rotated_index];
        else if (rotation_deltas_shape[1] == static_cast<size_t>(block_size))
            trig_index = rotation_deltas[rotated_index * block_size + token_offset];
    }
    return trig_index;
}

/*
 * Check if rotation should be applied.
 * Returns true if rotated_block_indices is not null, the number of rotated blocks is > 0,
 * and the block_id is found in the rotated list.
 */
bool should_rotate(int32_t block_id,
                   const int32_t* rotated_block_indices,
                   const ov::Shape& rotated_block_indices_shape,
                   int32_t& rotated_index) {
    if (!rotated_block_indices || rotated_block_indices_shape.empty()) {
        return false;
    }

    auto num_rotated_blocks = static_cast<int32_t>(rotated_block_indices_shape[0]);
    for (int32_t i = 0; i < num_rotated_blocks; i++) {
        if (rotated_block_indices[i] == block_id) {
            rotated_index = i;
            return true;
        }
    }
    return false;
}
}  // namespace paged_attention_utils

template <typename T>
void paged_attention(
    T* out,                               // output: attention result
    T* score,                             // output: concatenated raw scores
    const T* query,                       // shape: [batch_tokens, num_q_heads * head_size]
    const T* key,                         // shape: [batch_tokens, num_k_heads * head_size]
    const T* value,                       // shape: [batch_tokens, num_v_heads * head_size]
    T* key_cache,                         // shape: [num_blocks, num_k_heads, block_size, head_size]
    T* value_cache,                       // shape: [num_blocks, num_v_heads, block_size, head_size]
    const Shape& q_shape,                 // e.g. {batch_tokens, num_q_heads * head_size}
    const Shape& key_shape,               // e.g. {batch_tokens, num_k_heads * head_size}
    const Shape& value_shape,             // e.g. {batch_tokens, num_v_heads * head_size}
    const Shape& key_cache_shape,         // e.g. {num_blocks, num_k_heads, block_size, head_size}
    const Shape& value_cache_shape,       // e.g. {num_blocks, num_v_heads, block_size, head_size}
    const int32_t* past_lens,             // [batch_seq]: past tokens per sequence
    const Shape& past_lens_shape,         // e.g. {batch_seq}
    const int32_t* subsequence_begins,    // [batch_seq + 1]: start indices of new tokens per sequence
    const int32_t* block_indices,         // [num_blocks]: block table for each sequence
    const int32_t* block_indices_begins,  // [batch_seq + 1]: indices into block_indices per sequence
    const T* scale_ptr,                   // (Optional) attention scale factor (can be nullptr; default = 1)
    const int32_t* sliding_window_ptr,    // (Optional) sliding window parameter (can be nullptr; default = 0)
    const T* alibi_slopes,               // (Optional) [num_k_heads]: per-head bias slopes (can be nullptr; default = 0)
    const int32_t* max_context_len_ptr,  // max context length (for score output indexing)

    // Rotation parameters (if any is nullptr, rotation is skipped)
    const int32_t* rotated_block_indices,  // (Optional) [num_rotated_blocks]: blocks to which RoPE is applied
    const int32_t* rotation_deltas,  // (Optional) [num_rotated_blocks, block_size || 1]: indices into the trig LUT
    const T* rotation_trig_lut,      // (Optional) LUT: [lut_rows, head_size] (first half: cosines, second half: sines)
    const Shape& rotated_block_indices_shape,  // shape of rotated_block_indices (e.g. {num_rotated_blocks})
    const Shape& rotation_deltas_shape,        // shape of rotation_deltas (e.g. {num_rotated_blocks, block_size} or
                                               // {num_rotated_blocks, 1})
    const Shape& rotation_trig_lut_shape) {    // shape of rotation_trig_lut

    // Use default values for optional parameters if pointers are null.
    T scale = scale_ptr ? scale_ptr[0] : T(1);
    int32_t sliding_window = sliding_window_ptr ? sliding_window_ptr[0] : 0;
    int32_t max_context_len = max_context_len_ptr ? max_context_len_ptr[0] : 0;

    // Determine dimensions.
    int32_t num_blocks = key_cache_shape[0];
    int32_t num_k_heads = key_cache_shape[1];
    int32_t num_v_heads = value_cache_shape[1];
    int32_t block_size = key_cache_shape[2];
    int32_t head_size = key_cache_shape[3];

    int32_t batch_tokens = q_shape[0];
    int32_t query_features = q_shape[1];  // equals num_q_heads * head_size.
    int32_t num_q_heads = query_features / head_size;
    int32_t batch_seq = !past_lens_shape.empty() ? past_lens_shape[0] : 1;

    // Process each query token.
    for (int32_t token_idx = 0; token_idx < batch_tokens; token_idx++) {
        // Determine sequence index using subsequence_begins if available.
        int32_t seq_idx = 0;
        if (batch_seq > 1 && subsequence_begins) {
            for (int32_t s = 0; s < batch_seq; s++) {
                if (token_idx >= subsequence_begins[s] && token_idx < subsequence_begins[s + 1]) {
                    seq_idx = s;
                    break;
                }
            }
        }

        // --- Copy new token key and value vectors into cache ---
        // If this token is new (i.e. not part of the past), copy its key and value.
        if (subsequence_begins && token_idx >= subsequence_begins[seq_idx]) {
            int32_t new_token_offset = token_idx - subsequence_begins[seq_idx];
            // For simplicity, we assume new tokens are stored in the last block.
            int32_t block_id_new = num_blocks - 1;
            // Copy keys for each key head.
            for (int32_t kh = 0; kh < num_k_heads; kh++) {
                int32_t cache_idx = (((block_id_new * num_k_heads + kh) * block_size + new_token_offset) * head_size);
                const T* src_key = key + token_idx * key_shape[1] + kh * head_size;
                std::memcpy(key_cache + cache_idx, src_key, head_size * sizeof(T));
            }
            // Copy values for each value head.
            for (int32_t vh = 0; vh < num_v_heads; vh++) {
                int32_t cache_idx = (((block_id_new * num_v_heads + vh) * block_size + new_token_offset) * head_size);
                const T* src_val = value + token_idx * value_shape[1] + vh * head_size;
                std::memcpy(value_cache + cache_idx, src_val, head_size * sizeof(T));
            }
        }

        // Process each query head.
        for (int32_t h = 0; h < num_q_heads; h++) {
            const T* q_vec = query + token_idx * query_features + h * head_size;

            int32_t seq_new_tokens =
                subsequence_begins ? (subsequence_begins[seq_idx + 1] - subsequence_begins[seq_idx]) : 0;
            int32_t seq_past_tokens = past_lens ? past_lens[seq_idx] : 0;
            int32_t total_keys = seq_past_tokens + seq_new_tokens;
            std::vector<T> scores(total_keys, T(0));

            // Compute raw attention scores.
            for (int32_t k = 0; k < total_keys; k++) {
                T score_val = T(0);
                if (k < seq_past_tokens) {
                    // Retrieve key from cache.
                    int32_t block_id = -1, token_offset = 0;
                    bool found = paged_attention_utils::find_cached_token(seq_idx,
                                                                          k,
                                                                          block_indices,
                                                                          block_indices_begins,
                                                                          num_blocks,
                                                                          block_size,
                                                                          block_id,
                                                                          token_offset);
                    if (!found)
                        continue;

                    // If token falls within the sliding window of the first block, set score to -∞.
                    int32_t first_block_for_seq =
                        (block_indices_begins ? block_indices[block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq && token_offset < sliding_window) {
                        score_val = -std::numeric_limits<T>::infinity();
                    } else {
                        int32_t k_head = h / (num_q_heads / num_k_heads);
                        int32_t cache_idx =
                            (((block_id * num_k_heads + k_head) * block_size + token_offset) * head_size);
                        const T* key_vec = key_cache + cache_idx;

                        // Check for RoPE adjustment.
                        bool do_rotate = false;
                        int32_t rotated_index = -1;
                        if (rotated_block_indices && rotation_deltas && rotation_trig_lut) {
                            do_rotate = paged_attention_utils::should_rotate(block_id,
                                                                             rotated_block_indices,
                                                                             rotated_block_indices_shape,
                                                                             rotated_index);
                        }
                        if (do_rotate) {
                            int32_t trig_index = paged_attention_utils::get_trig_index(rotation_deltas,
                                                                                       rotation_deltas_shape,
                                                                                       rotated_index,
                                                                                       token_offset,
                                                                                       block_size);
                            std::vector<T> temp_key(key_vec, key_vec + head_size);
                            paged_attention_utils::apply_rope(temp_key.data(),
                                                              head_size,
                                                              rotation_trig_lut,
                                                              trig_index);
                            score_val = paged_attention_utils::dot_product(q_vec, temp_key.data(), head_size);
                        } else {
                            score_val = paged_attention_utils::dot_product(q_vec, key_vec, head_size);
                        }
                    }
                } else {
                    // Retrieve key from new input.
                    int32_t new_token_idx = subsequence_begins ? (subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                                               : (k - seq_past_tokens);
                    int32_t k_head = h % num_k_heads;
                    const T* key_vec = key + new_token_idx * key_shape[1] + k_head * head_size;
                    score_val = paged_attention_utils::dot_product(q_vec, key_vec, head_size);
                }
                // Apply scale and add alibi bias according to the formula:
                // softmax(q * transpose(k) + m · [–(i – 1), …, –2, –1, 0])
                T alibi = alibi_slopes ? alibi_slopes[h % num_k_heads] : T(0);
                score_val *= scale;
                score_val += alibi * static_cast<T>(-(total_keys - k - 1));
                scores[k] = score_val;
            }

            paged_attention_utils::softmax(scores);

            // Compute weighted sum over value vectors.
            std::vector<T> out_vec(head_size, T(0));
            for (int32_t k = 0; k < total_keys; k++) {
                T weight = scores[k];
                if (k < seq_past_tokens) {
                    int32_t block_id = -1, token_offset = 0;
                    bool found = paged_attention_utils::find_cached_token(seq_idx,
                                                                          k,
                                                                          block_indices,
                                                                          block_indices_begins,
                                                                          num_blocks,
                                                                          block_size,
                                                                          block_id,
                                                                          token_offset);
                    if (!found)
                        continue;
                    // Skip accumulation for tokens in the sliding window region.
                    int32_t first_block_for_seq =
                        (block_indices_begins ? block_indices[block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq && token_offset < sliding_window)
                        continue;

                    int32_t v_head = h % num_v_heads;
                    int32_t cache_idx = (((block_id * num_v_heads + v_head) * block_size + token_offset) * head_size);
                    const T* raw_val_vec = value_cache + cache_idx;

                    bool do_rotate = false;
                    int32_t rotated_index = -1;
                    if (rotated_block_indices && rotation_deltas && rotation_trig_lut) {
                        do_rotate = paged_attention_utils::should_rotate(block_id,
                                                                         rotated_block_indices,
                                                                         rotated_block_indices_shape,
                                                                         rotated_index);
                    }
                    if (do_rotate) {
                        int32_t trig_index = paged_attention_utils::get_trig_index(rotation_deltas,
                                                                                   rotation_deltas_shape,
                                                                                   rotated_index,
                                                                                   token_offset,
                                                                                   block_size);
                        std::vector<T> temp_value(raw_val_vec, raw_val_vec + head_size);
                        paged_attention_utils::apply_rope(temp_value.data(), head_size, rotation_trig_lut, trig_index);
                        for (int32_t d = 0; d < head_size; d++) {
                            out_vec[d] += weight * temp_value[d];
                        }
                    } else {
                        for (int32_t d = 0; d < head_size; d++) {
                            out_vec[d] += weight * raw_val_vec[d];
                        }
                    }
                } else {
                    int32_t new_token_idx = subsequence_begins ? (subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                                               : (k - seq_past_tokens);
                    int32_t v_head = h % num_v_heads;
                    const T* val_vec = value + new_token_idx * value_shape[1] + v_head * head_size;
                    for (int32_t d = 0; d < head_size; d++) {
                        out_vec[d] += weight * val_vec[d];
                    }
                }
                // Write the raw score into the score output.
                int32_t global_score_index = seq_idx * max_context_len + k;
                score[global_score_index] = scores[k];
            }
            // Write the computed attention result for this query token and head.
            T* dst = out + token_idx * query_features + h * head_size;
            std::memcpy(dst, out_vec.data(), head_size * sizeof(T));
        }
    }
}

}  // namespace ov::reference
