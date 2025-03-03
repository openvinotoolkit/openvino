// Copyright (C) 2018-2024 Intel Corporation
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
    const int32_t* subsequence_begins,     // [batch_size_in_sequences + 1]: start indices of new tokens per sequence
    const int32_t* block_indices,          // [num_blocks]: block table for each sequence
    const int32_t* block_indices_begins,   // [batch_size_in_sequences + 1]: indices into block_indices per sequence
    const T* scale,                        // attention scale factor (scalar)
    const int32_t* sliding_window,         // sliding window parameter (scalar)
    const T* alibi_slopes,                 // [num_kv_heads]: per-head bias slopes
    const int32_t* max_context_len,        // max context length (for score output indexing) (scalar)
    const int32_t* rotated_block_indices,  // [num_rotated_blocks]: blocks to which RoPE is applied
    const int32_t* rotation_deltas,        // [num_rotated_blocks, block_size || 1]: indices into the trig LUT
    const int32_t* rotation_trig_lut,      // LUT: [lut_rows, head_size] (first half: cosines, second half: sines)
    const Shape& rotated_block_indices_shape,  // shape of rotated_block_indices (e.g. {num_rotated_blocks})
    const Shape& rotation_deltas_shape,        // shape of rotation_deltas (e.g. {num_rotated_blocks, block_size} or
                                               // {num_rotated_blocks, 1})
    const Shape& rotation_trig_lut_shape);     // shape of rotation_trig_lut

}  // namespace reference
}  // namespace ov
