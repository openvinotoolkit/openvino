// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace reference {
namespace paged_attention_utils {

// --- Helper / Unit Functions ---

template <typename T>
T dot_product(const T* a, const T* b, int32_t size);

template <typename T>
void softmax(std::vector<T>& scores);

/*
 * Apply RoPE (Rotary Positional Embedding) rotation to a vector.
 */
template <typename T>
void apply_rope(T* vec, int32_t head_size, const T* rotation_trig_lut, int32_t trig_index);

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
                       int32_t& token_offset);
/*
 * Given rotation_deltas and its shape, compute the trig index for RoPE.
 * Returns 0 if rotation parameters are not available.
 */
int32_t get_trig_index(const int32_t* rotation_deltas,
                       const ov::Shape& rotation_deltas_shape,
                       int32_t rotated_index,
                       int32_t token_offset,
                       int32_t block_size);
/*
 * Check if rotation should be applied.
 * Returns true if rotated_block_indices is not null, the number of rotated blocks is > 0,
 * and the block_id is found in the rotated list.
 */
bool should_rotate(int32_t block_id,
                   const int32_t* rotated_block_indices,
                   const ov::Shape& rotated_block_indices_shape,
                   int32_t& rotated_index);

}  // namespace paged_attention_utils

template <typename T>
void paged_attention(
    T* out,                                // first output: attention result
    T* score,                              // second output: concatenated raw scores
    const T* query,                        // shape: [batch_tokens, num_heads * head_size]
    const T* key,                          // shape: [batch_tokens, num_kv_heads * head_size]
    const T* value,                        // shape: [batch_tokens, num_kv_heads * head_size]
    const T* key_cache,                    // shape: [num_blocks, num_kv_heads, block_size, head_size]
    const T* value_cache,                  // shape: [num_blocks, num_kv_heads, block_size, head_size]
    const Shape& q_shape,                  // e.g. {batch_tokens, num_heads * head_size}
    const Shape& kv_shape,                 // e.g. {batch_tokens, num_kv_heads * head_size}
    const Shape& kv_cache_shape,           // e.g. {num_blocks, num_kv_heads, block_size, head_size}
    const int32_t* past_lens,              // [batch_size_in_sequences]: past tokens per sequence
    const Shape& past_lens_shape,          // e.g. {batch_seq}
    const int32_t* subsequence_begins,     // [batch_size_in_sequences + 1]: start indices of new tokens per sequence
    const int32_t* block_indices,          // [num_blocks]: block table for each sequence
    const int32_t* block_indices_begins,   // [batch_size_in_sequences + 1]: indices into block_indices per sequence
    const T* scale,                        // (optional) attention scale factor (scalar)
    const int32_t* sliding_window,         // (optional) sliding window parameter (scalar)
    const T* alibi_slopes,                 // (optional) [num_kv_heads]: per-head bias slopes
    const int32_t* max_context_len,        // max context length (for score output indexing) (scalar)
    const int32_t* rotated_block_indices,  // (optional) [num_rotated_blocks]: blocks to which RoPE is applied
    const int32_t* rotation_deltas,  // (optional) [num_rotated_blocks, block_size || 1]: indices into the trig LUT
    const T* rotation_trig_lut,      // (optional) LUT: [lut_rows, head_size] (first half: cosines, second half: sines)
    const Shape& rotated_block_indices_shape,  // shape of rotated_block_indices (e.g. {num_rotated_blocks})
    const Shape& rotation_deltas_shape,        // shape of rotation_deltas (e.g. {num_rotated_blocks, block_size} or
                                               // {num_rotated_blocks, 1})
    const Shape& rotation_trig_lut_shape);     // shape of rotation_trig_lut

}  // namespace reference
}  // namespace ov
