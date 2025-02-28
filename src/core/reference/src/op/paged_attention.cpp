// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/paged_attention.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "openvino/core/shape.hpp"

// ==== HELPER FUNCS ====
template <typename T>
T dot_product(const T* a, const T* b, int32_t size) {
    T sum = T(0);
    for (int32_t i = 0; i < size; i++)
        sum += a[i] * b[i];
    return sum;
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
    // Each row is of length head_size.
    const float* row = rotation_trig_lut + trig_index * head_size;
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
 * Helper to check whether a given block_id is present in rotated_block_indices.
 * If found, returns true and sets rotated_index to the position within rotated_block_indices.
 */
bool get_rotated_index(int32_t block_id,
                       const int32_t* rotated_block_indices,
                       int32_t num_rotated_blocks,
                       int32_t& rotated_index) {
    for (int32_t i = 0; i < num_rotated_blocks; i++) {
        if (rotated_block_indices[i] == block_id) {
            rotated_index = i;
            return true;
        }
    }
    return false;
}

namespace ov {
namespace reference {

template <typename T>
void paged_attention(
    T* out,                                    // output: attention result
    T* score,                                  // output: concatenated raw scores
    const T* query,                            // shape: [batch_tokens, num_heads * head_size]
    const T* key,                              // shape: [batch_tokens, num_kv_heads * head_size]
    const T* value,                            // shape: [batch_tokens, num_kv_heads * head_size]
    const T* key_cache,                        // shape: [num_blocks, num_kv_heads, block_size, head_size]
    const T* value_cache,                      // shape: [num_blocks, num_kv_heads, block_size, head_size]
    const Shape& q_shape,                      // e.g. {batch_tokens, num_heads * head_size}
    const Shape& kv_shape,                     // e.g. {batch_tokens, num_kv_heads * head_size}
    const Shape& kv_cache_shape,               // e.g. {num_blocks, num_kv_heads, block_size, head_size}
    const int32_t* past_lens,                  // [batch_seq]: past tokens per sequence
    const Shape& past_lens_shape,              // e.g. {batch_seq}
    const int32_t* subsequence_begins,         // [batch_seq + 1]: start indices of new tokens per sequence
    const int32_t* block_indices,              // [num_blocks]: block table for each sequence
    const int32_t* block_indices_begins,       // [batch_seq + 1]: indices into block_indices per sequence
    const T* scale_ptr,                        // attention scale factor
    const int32_t* sliding_window_ptr,         // sliding window parameter
    const T* alibi_slopes,                     // [num_kv_heads]: per-head bias slopes
    const int32_t* max_context_len_ptr,        // max context length (for score output indexing)
    const int32_t* rotated_block_indices,      // [num_rotated_blocks]: blocks to which RoPE is applied
    const int32_t* rotation_deltas,            // [num_rotated_blocks, block_size || 1]: indices into the trig LUT
    const float* rotation_trig_lut,            // LUT: [lut_rows, head_size] (first half: cosines, second half: sines)
    const Shape& rotated_block_indices_shape,  // shape of rotated_block_indices (e.g. {num_rotated_blocks})
    const Shape& rotation_deltas_shape,        // shape of rotation_deltas (e.g. {num_rotated_blocks, block_size} or
                                               // {num_rotated_blocks, 1})
    const Shape& rotation_trig_lut_shape) {    // shape of rotation_trig_lut

    T scale = scale_ptr[0];
    int sliding_window = sliding_window_ptr[0];
    int max_context_len = max_context_len_ptr[0];

    // Determine dimensions.
    int num_blocks = kv_cache_shape[0];
    int num_kv_heads = kv_cache_shape[1];
    int block_size = kv_cache_shape[2];
    int head_size = kv_cache_shape[3];

    int batch_tokens = q_shape[0];
    int query_features = q_shape[1];  // equals num_heads * head_size.
    int num_heads = query_features / head_size;
    int batch_seq = past_lens_shape[0];

    // Process each query token.
    for (int token_idx = 0; token_idx < batch_tokens; token_idx++) {
        // Determine sequence index.
        int seq_idx = 0;
        if (batch_seq > 1 && subsequence_begins) {
            for (int s = 0; s < batch_seq; s++) {
                if (token_idx >= subsequence_begins[s] && token_idx < subsequence_begins[s + 1]) {
                    seq_idx = s;
                    break;
                }
            }
        }
        // Process each query head.
        for (int h = 0; h < num_heads; h++) {
            const T* q_vec = query + token_idx * query_features + h * head_size;

            int seq_new_tokens =
                subsequence_begins ? (subsequence_begins[seq_idx + 1] - subsequence_begins[seq_idx]) : 0;
            int seq_past_tokens = past_lens ? past_lens[seq_idx] : 0;
            int total_keys = seq_past_tokens + seq_new_tokens;
            std::vector<T> scores(total_keys, T(0));

            // Compute raw attention scores.
            for (int k = 0; k < total_keys; k++) {
                T score_val = T(0);
                if (k < seq_past_tokens) {
                    // Retrieve key from cache.
                    int block_start = block_indices_begins ? block_indices_begins[seq_idx] : 0;
                    int block_end = block_indices_begins ? block_indices_begins[seq_idx + 1] : num_blocks;
                    int remaining = k;
                    int block_id = -1;
                    int token_offset = 0;
                    for (int b = block_start; b < block_end; b++) {
                        if (remaining < block_size) {
                            block_id = block_indices[b];
                            token_offset = remaining;
                            break;
                        }
                        remaining -= block_size;
                    }
                    if (block_id < 0)
                        continue;
                    // Determine if this token falls in the sliding window of the first block.
                    int first_block_for_seq =
                        (block_indices_begins ? block_indices[block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq && token_offset < sliding_window) {
                        score_val = -std::numeric_limits<T>::infinity();
                    } else {
                        int kv_head = h % num_kv_heads;
                        const T* key_vec =
                            key_cache + (((block_id * num_kv_heads + kv_head) * block_size + token_offset) * head_size);
                        // Check for RoPE adjustment.
                        bool do_rotate = false;
                        int rotated_index = -1;
                        int num_rotated_blocks = rotated_block_indices_shape[0];
                        if (rotated_block_indices && num_rotated_blocks > 0) {
                            if (get_rotated_index(block_id, rotated_block_indices, num_rotated_blocks, rotated_index))
                                do_rotate = true;
                        }
                        if (do_rotate) {
                            int trig_index = 0;
                            if (rotation_deltas_shape.size() >= 2 && rotation_deltas_shape[1] == 1)
                                trig_index = rotation_deltas[rotated_index];
                            else if (rotation_deltas_shape.size() >= 2 && rotation_deltas_shape[1] == block_size)
                                trig_index = rotation_deltas[rotated_index * block_size + token_offset];
                            std::vector<T> temp_key(key_vec, key_vec + head_size);
                            apply_rope(temp_key.data(), head_size, rotation_trig_lut, trig_index);
                            score_val = dot_product(q_vec, temp_key.data(), head_size);
                        } else {
                            score_val = dot_product(q_vec, key_vec, head_size);
                        }
                    }
                } else {
                    // Retrieve key from new input.
                    int new_token_idx = subsequence_begins ? (subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                                           : (k - seq_past_tokens);
                    int kv_head = h % num_kv_heads;
                    const T* key_vec = key + new_token_idx * kv_shape[1] + kv_head * head_size;
                    score_val = dot_product(q_vec, key_vec, head_size);
                }
                // Scale and add alibi bias (indexed by kv head).
                score_val *= static_cast<T>(scale);
                score_val += alibi_slopes[(h % num_kv_heads)] * static_cast<T>(k);
                scores[k] = score_val;
            }

            softmax(scores);

            // Compute weighted sum over value vectors.
            std::vector<T> out_vec(head_size, T(0));
            for (int k = 0; k < total_keys; k++) {
                T weight = scores[k];
                if (k < seq_past_tokens) {
                    int block_start = block_indices_begins ? block_indices_begins[seq_idx] : 0;
                    int block_end = block_indices_begins ? block_indices_begins[seq_idx + 1] : num_blocks;
                    int remaining = k;
                    int block_id = -1;
                    int token_offset = 0;
                    for (int b = block_start; b < block_end; b++) {
                        if (remaining < block_size) {
                            block_id = block_indices[b];
                            token_offset = remaining;
                            break;
                        }
                        remaining -= block_size;
                    }
                    if (block_id < 0)
                        continue;
                    // If token is in the sliding window region, skip accumulation.
                    int first_block_for_seq =
                        (block_indices_begins ? block_indices[block_indices_begins[seq_idx]] : -1);
                    if (first_block_for_seq >= 0 && block_id == first_block_for_seq && token_offset < sliding_window)
                        continue;

                    int kv_head = h % num_kv_heads;
                    const T* raw_val_vec =
                        value_cache + (((block_id * num_kv_heads + kv_head) * block_size + token_offset) * head_size);

                    bool do_rotate = false;
                    int rotated_index = -1;
                    int num_rotated_blocks = rotated_block_indices_shape[0];
                    if (rotated_block_indices && num_rotated_blocks > 0) {
                        if (get_rotated_index(block_id, rotated_block_indices, num_rotated_blocks, rotated_index))
                            do_rotate = true;
                    }
                    if (do_rotate) {
                        int trig_index = 0;
                        if (rotation_deltas_shape.size() >= 2 && rotation_deltas_shape[1] == 1)
                            trig_index = rotation_deltas[rotated_index];
                        else if (rotation_deltas_shape.size() >= 2 && rotation_deltas_shape[1] == block_size)
                            trig_index = rotation_deltas[rotated_index * block_size + token_offset];
                        std::vector<T> temp_value(raw_val_vec, raw_val_vec + head_size);
                        apply_rope(temp_value.data(), head_size, rotation_trig_lut, trig_index);
                        for (int d = 0; d < head_size; d++) {
                            out_vec[d] += weight * temp_value[d];
                        }
                    } else {
                        for (int d = 0; d < head_size; d++) {
                            out_vec[d] += weight * raw_val_vec[d];
                        }
                    }
                } else {
                    int new_token_idx = subsequence_begins ? (subsequence_begins[seq_idx] + (k - seq_past_tokens))
                                                           : (k - seq_past_tokens);
                    int kv_head = h % num_kv_heads;
                    const T* val_vec = value + new_token_idx * kv_shape[1] + kv_head * head_size;
                    for (int d = 0; d < head_size; d++) {
                        out_vec[d] += weight * val_vec[d];
                    }
                }
                // Write the raw score into the score output.
                int global_score_index = seq_idx * max_context_len + k;
                score[global_score_index] = scores[k];
            }
            // Write the computed attention result for this query token and head.
            T* dst = out + token_idx * query_features + h * head_size;
            std::memcpy(dst, out_vec.data(), head_size * sizeof(T));
        }
    }
}
}  // namespace reference
}  // namespace ov
