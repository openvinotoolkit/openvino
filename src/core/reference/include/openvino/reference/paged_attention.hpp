// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov::reference {
namespace paged_attention_utils {

// TODO Delete when redundant
//---------------------------------------------------------------------------
// Debug-print utilities
//---------------------------------------------------------------------------

// Recursive helper for printing an N-dimensional tensor
template <typename T>
void debug_print_tensor_recursive(const T* data,
                                  const std::vector<size_t>& shape,
                                  size_t dim,
                                  size_t offset,
                                  const std::string& prefix,
                                  int indent) {
    if (dim == shape.size() - 1) {
        // Last dimension: print a flat row
        std::cout << std::string(indent, ' ') << "[ ";
        for (size_t i = 0; i < shape[dim]; ++i) {
            std::cout << std::fixed << std::setprecision(6) << data[offset + i];
            if (i + 1 < shape[dim])
                std::cout << ", ";
        }
        std::cout << " ]";
    } else {
        std::cout << std::string(indent, ' ') << "[\n";
        size_t stride = 1;
        for (size_t d = dim + 1; d < shape.size(); ++d) {
            stride *= shape[d];
        }
        for (size_t i = 0; i < shape[dim]; ++i) {
            debug_print_tensor_recursive(data, shape, dim + 1, offset + i * stride, prefix, indent + 2);
            if (i + 1 < shape[dim])
                std::cout << ",\n";
            else
                std::cout << "\n";
        }
        std::cout << std::string(indent, ' ') << "]";
    }
}

template <typename T>
void debug_print_tensor(const T* data, const std::vector<size_t>& shape, const std::string& name) {
    // Print header: name and shape
    std::cout << name << " (shape = [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << "]):\n";
    // Recursively print contents
    debug_print_tensor_recursive(data, shape, 0, 0, name, 0);
    std::cout << "\n\n";
}

template <typename T>
void debug_print_vector(const T* data, size_t length, const std::string& name) {
    std::cout << name << " (length = " << length << "): [ ";
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < length; ++i) {
        std::cout << data[i];
        if (i + 1 < length)
            std::cout << ", ";
    }
    std::cout << " ]\n\n";
}

template <typename T>
void debug_print_std_vector(const std::vector<T>& vec, const std::string& name) {
    debug_print_vector(vec.data(), vec.size(), name);
}

template <typename T>
void debug_print_scalar(const T& value, const std::string& name) {
    std::cout << name << ": " << std::fixed << std::setprecision(6) << value << "\n\n";
}

inline void debug_print_int_scalar(const int32_t& value, const std::string& name) {
    std::cout << name << ": " << value << "\n\n";
}

inline void debug_print_size_t_scalar(const size_t& value, const std::string& name) {
    std::cout << name << ": " << value << "\n\n";
}

//---------------------------------------------------------------------------
// Context: Holds all parameters and computed dimensions in one place.
//---------------------------------------------------------------------------
struct PagedAttentionContext {
    // Attention with cache inputs ptrs
    const void* query;
    const void* key;
    const void* value;
    void* key_cache;
    void* value_cache;

    // Sequencing related input ptrs
    const int32_t* past_lens;
    const int32_t* subsequence_begins;
    int32_t* block_indices;
    int32_t* block_indices_begins;
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

int32_t find_first_negative(const int32_t* data, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < 0) {
            return int32_t(i);
        }
    }
    // When not found returns negative idx
    return -1;
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
    int32_t begin = ctx.block_indices_begins[sequence_index];
    int32_t count = ctx.sequence_block_count[sequence_index];
    int32_t maxb = std::numeric_limits<int32_t>::max();

    if (count < maxb) {
        int32_t free_block = find_first_negative(ctx.block_indices, ctx.num_blocks);
        if (free_block != -1) {
            ctx.sequence_block_count[sequence_index] = count + 1;
            return free_block;
        }
    }
    int32_t oldest = ctx.block_indices[begin];
    for (int32_t i = 0; i + 1 < count; ++i) {
        ctx.block_indices[begin + i] = ctx.block_indices[begin + i + 1];
    }
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

    int32_t begin = ctx.block_indices_begins[sequence_index];
    ctx.block_indices[begin + ctx.sequence_block_count[sequence_index] - 1] = block_id;

    // Debug: report insertion parameters
    std::cout << "[Debug] insert_new_token_into_cache:\n";
    std::cout << "  token_index = " << token_index << "\n";
    std::cout << "  sequence_index = " << sequence_index << "\n";
    std::cout << "  local_index = " << local_index << "\n";
    std::cout << "  offset = " << offset << "\n";
    std::cout << "  block_id = " << block_id << "\n\n";

    // Copy key and value vectors, with debug prints
    for (size_t head = 0; head < ctx.num_heads; ++head) {
        size_t k_offset = ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.key_head_size;
        const T* src_key = key_ptr + token_index * ctx.key_feature_size + head * ctx.key_head_size;
        std::memcpy(key_cache_ptr + k_offset, src_key, ctx.key_head_size * sizeof(T));

        size_t v_offset = ((block_id * ctx.num_heads + head) * ctx.block_size + offset) * ctx.value_head_size;
        const T* src_val = value_ptr + token_index * ctx.value_feature_size + head * ctx.value_head_size;
        std::memcpy(value_cache_ptr + v_offset, src_val, ctx.value_head_size * sizeof(T));

        // Debug: show inserted key slice
        std::vector<size_t> key_slice_shape = {ctx.key_head_size};
        debug_print_tensor(src_key,
                           key_slice_shape,
                           "  key_cache insertion [block=" + std::to_string(block_id) +
                               ", head=" + std::to_string(head) + ", offset=" + std::to_string(offset) + "]");

        // Debug: show inserted value slice
        std::vector<size_t> val_slice_shape = {ctx.value_head_size};
        debug_print_tensor(src_val,
                           val_slice_shape,
                           "  value_cache insertion [block=" + std::to_string(block_id) +
                               ", head=" + std::to_string(head) + ", offset=" + std::to_string(offset) + "]");
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
    int32_t begin = ctx.block_indices_begins[sequence_index];
    int32_t end = ctx.block_indices_begins[sequence_index + 1];
    int32_t count = end - begin;
    if (token_position < count) {
        block_id = ctx.block_indices[begin + token_position];
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

    // Debug: print the key vector used for dot product
    std::vector<size_t> key_vec_shape = {ctx.key_head_size};
    debug_print_tensor(key_vec,
                       key_vec_shape,
                       "    cached key_vec [block=" + std::to_string(block_id) + ", head=" + std::to_string(head) +
                           ", offset=" + std::to_string(offset) + "]");

    T score = dot_product(query_vec, key_vec, ctx.key_head_size);
    int32_t rot_idx;
    if (should_apply_rotation(block_id, ctx, rot_idx)) {
        std::vector<T> tmp(key_vec, key_vec + ctx.key_head_size);
        apply_rope(tmp.data(),
                   ctx.key_head_size,
                   static_cast<const T*>(ctx.rotation_trig_lut),
                   get_trig_index(rot_idx, offset, ctx));
        score = dot_product(query_vec, tmp.data(), ctx.key_head_size);

        // Debug: print the rotated key vector if rotation applied
        debug_print_std_vector(tmp,
                               "    rotated key_vec [block=" + std::to_string(block_id) +
                                   ", head=" + std::to_string(head) + ", offset=" + std::to_string(offset) + "]");
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

    // Debug: print the new key vector used for dot product
    std::vector<size_t> new_key_vec_shape = {ctx.key_head_size};
    debug_print_tensor(
        key_vec,
        new_key_vec_shape,
        "    new key_vec [token=" + std::to_string(new_token_idx) + ", head=" + std::to_string(head) + "]");

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

    // Debug: print the value vector used for accumulation
    std::vector<size_t> val_vec_shape = {ctx.value_head_size};
    debug_print_tensor(val_vec,
                       val_vec_shape,
                       "    cached value_vec [block=" + std::to_string(block_id) + ", head=" + std::to_string(head) +
                           ", offset=" + std::to_string(offset) + "]");

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

    // Debug: print the new value vector used for accumulation
    std::vector<size_t> new_val_vec_shape = {ctx.value_head_size};
    debug_print_tensor(
        val_vec,
        new_val_vec_shape,
        "    new value_vec [token=" + std::to_string(new_token_idx) + ", head=" + std::to_string(head) + "]");

    for (size_t i = 0; i < ctx.value_head_size; ++i) {
        output[i] += weight * val_vec[i];
    }
}

}  // namespace paged_attention_utils

//---------------------------------------------------------------------------
// Main PagedAttention function
//---------------------------------------------------------------------------
template <typename T>
void paged_attention(T* out,    // Output attention result
                     T* score,  // Output concatenated raw scores

                     const T* query,                      // [batch_tokens, num_heads * q_head_size]
                     const T* key,                        // [batch_tokens, num_heads * k_head_size]
                     const T* value,                      // [batch_tokens, num_heads * v_head_size]
                     T* key_cache,                        // [num_blocks, num_heads, block_size, k_head_size]
                     T* value_cache,                      // [num_blocks, num_heads, block_size, v_head_size]
                     const int32_t* past_lens,            // [batch_seq]
                     const int32_t* subsequence_begins,   // [batch_seq+1]
                     int32_t* block_indices,              // [num_blocks]
                     int32_t* block_indices_begins,       // [batch_seq+1]
                     const T* scale_ptr,                  // (Optional) scale factor
                     const int32_t* sliding_window_ptr,   // (Optional)
                     const T* alibi_slopes,               // (Optional) [num_heads]
                     const int32_t* max_context_len_ptr,  // (Optional)

                     const int32_t* rotated_block_indices,  // (Optional) [num_rotated_blocks]
                     const int32_t* rotation_deltas,        // (Optional) [num_rotated_blocks × rotation_deltas_dim]
                     const T* rotation_trig_lut,            // (Optional) [rotation_lut_rows, head_size]

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

    //--------------------------------------------------------------------------------
    // Initial Debug Print of Raw Inputs
    //--------------------------------------------------------------------------------

    std::cout << "===== paged_attention: Initial Inputs =====\n\n";

    // Query
    {
        std::vector<size_t> q_shape_vec = {query_shape[0], query_shape[1]};
        debug_print_tensor(query, q_shape_vec, "Input Q");
    }
    // Key
    {
        std::vector<size_t> k_shape_vec = {key_shape[0], key_shape[1]};
        debug_print_tensor(key, k_shape_vec, "Input K");
    }
    // Value
    {
        std::vector<size_t> v_shape_vec = {value_shape[0], value_shape[1]};
        debug_print_tensor(value, v_shape_vec, "Input V");
    }
    // Key Cache (before any insertions)
    {
        std::vector<size_t> kc_shape_vec = {key_cache_shape[0],
                                            key_cache_shape[1],
                                            key_cache_shape[2],
                                            key_cache_shape[3]};
        debug_print_tensor(key_cache, kc_shape_vec, "Initial Key Cache");
    }
    // Value Cache (before any insertions)
    {
        std::vector<size_t> vc_shape_vec = {value_cache_shape[0],
                                            value_cache_shape[1],
                                            value_cache_shape[2],
                                            value_cache_shape[3]};
        debug_print_tensor(value_cache, vc_shape_vec, "Initial Value Cache");
    }
    // past_lens
    if (past_lens) {
        std::vector<size_t> pl_shape = {past_lens_shape[0]};
        debug_print_tensor(past_lens, pl_shape, "past_lens");
    }
    // subsequence_begins
    if (subsequence_begins) {
        std::vector<size_t> sb_shape = {past_lens_shape[0] + 1};
        debug_print_tensor(subsequence_begins, sb_shape, "subsequence_begins");
    }
    // block_indices
    if (block_indices) {
        // block_indices length = key_cache_shape[0] (num_blocks)
        std::vector<size_t> bi_shape = {key_cache_shape[0]};
        debug_print_tensor(block_indices, bi_shape, "block_indices (initial)");
    }
    // block_indices_begins
    if (block_indices_begins) {
        std::vector<size_t> bib_shape = {past_lens_shape[0] + 1};
        debug_print_tensor(block_indices_begins, bib_shape, "block_indices_begins");
    }
    // scale
    if (scale_ptr) {
        debug_print_scalar(scale_ptr[0], "scale_ptr[0]");
    }
    // sliding_window
    if (sliding_window_ptr) {
        debug_print_int_scalar(sliding_window_ptr[0], "sliding_window_ptr[0]");
    }
    // alibi_slopes
    if (alibi_slopes) {
        std::vector<size_t> al_shape = {query_shape[1] / (/* assume evenly divided among heads? */ 1)};
        // actual shape is [num_heads], but for debug we print the raw array length
        // (user can infer from context.num_heads below)
        debug_print_tensor(static_cast<const T*>(alibi_slopes), al_shape, "alibi_slopes");
    }
    // max_context_len
    if (max_context_len_ptr) {
        debug_print_int_scalar(max_context_len_ptr[0], "max_context_len_ptr[0]");
    }
    // rotated_block_indices
    if (rotated_block_indices) {
        std::vector<size_t> rbi_shape = {rotated_block_indices_shape[0]};
        debug_print_tensor(rotated_block_indices, rbi_shape, "rotated_block_indices");
    }
    // rotation_deltas
    if (rotation_deltas) {
        std::vector<size_t> rd_shape = {rotation_deltas_shape[0], rotation_deltas_shape[1]};
        debug_print_tensor(rotation_deltas, rd_shape, "rotation_deltas");
    }
    // rotation_trig_lut
    if (rotation_trig_lut) {
        std::vector<size_t> rtl_shape = {rotation_trig_lut_shape[0], rotation_trig_lut_shape[1]};
        debug_print_tensor(rotation_trig_lut, rtl_shape, "rotation_trig_lut");
    }

    std::cout << "===========================================\n\n";

    //--------------------------------------------------------------------------------
    // Build Context Print
    //--------------------------------------------------------------------------------

    PagedAttentionContext ctx{};

    // Assign inputs
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
    ctx.max_context_length = max_context_len_ptr ? max_context_len_ptr[0] : 0;
    ctx.sliding_window = sliding_window_ptr ? sliding_window_ptr[0] : 0;

    // Dimensions
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

    // Seed per-sequence counts
    ctx.sequence_block_count.resize(ctx.batch_sequence_count);
    for (size_t s = 0; s < ctx.batch_sequence_count; ++s) {
        int32_t begin = block_indices_begins[s];
        int32_t end = block_indices_begins[s + 1];
        ctx.sequence_block_count[s] = end - begin;
    }

    std::cout << "===== paged_attention: Context Contents =====\n\n";
    debug_print_size_t_scalar(ctx.batch_tokens, "ctx.batch_tokens");
    debug_print_size_t_scalar(ctx.batch_sequence_count, "ctx.batch_sequence_count");
    debug_print_size_t_scalar(ctx.num_heads, "ctx.num_heads");
    debug_print_size_t_scalar(ctx.block_size, "ctx.block_size");
    debug_print_size_t_scalar(ctx.num_blocks, "ctx.num_blocks");
    debug_print_size_t_scalar(ctx.query_head_size, "ctx.query_head_size");
    debug_print_size_t_scalar(ctx.key_head_size, "ctx.key_head_size");
    debug_print_size_t_scalar(ctx.value_head_size, "ctx.value_head_size");
    debug_print_size_t_scalar(ctx.query_feature_size, "ctx.query_feature_size");
    debug_print_size_t_scalar(ctx.key_feature_size, "ctx.key_feature_size");
    debug_print_size_t_scalar(ctx.value_feature_size, "ctx.value_feature_size");
    debug_print_int_scalar(ctx.max_context_length, "ctx.max_context_length");
    debug_print_int_scalar(ctx.sliding_window, "ctx.sliding_window");
    debug_print_size_t_scalar(ctx.num_rotated_blocks, "ctx.num_rotated_blocks");
    debug_print_size_t_scalar(ctx.rotation_lut_rows, "ctx.rotation_lut_rows");
    debug_print_size_t_scalar(ctx.rotation_deltas_dim, "ctx.rotation_deltas_dim");
    debug_print_std_vector(ctx.sequence_block_count, "ctx.sequence_block_count");
    std::cout << "==============================================\n\n";

    // Compute scale factor
    T scale = scale_ptr ? scale_ptr[0] : T(1) / std::sqrt(T(ctx.value_head_size));
    debug_print_scalar(scale, "Computed scale");

    //--------------------------------------------------------------------------------
    // Main Loop
    //--------------------------------------------------------------------------------

    for (size_t token_idx = 0; token_idx < ctx.batch_tokens; ++token_idx) {
        size_t seq_idx = get_sequence_index(token_idx, ctx);

        std::cout << "----- Processing token " << token_idx << " (sequence " << seq_idx << ") -----\n\n";

        // Update KV cache if new token in this sequence
        if (subsequence_begins && token_idx >= size_t(subsequence_begins[seq_idx])) {
            insert_new_token_into_cache<T>(ctx, token_idx, seq_idx);

            // Debug: print key_cache and value_cache after insertion
            {
                std::vector<size_t> kc_shape_vec = {key_cache_shape[0],
                                                    key_cache_shape[1],
                                                    key_cache_shape[2],
                                                    key_cache_shape[3]};
                debug_print_tensor(key_cache, kc_shape_vec, "Key Cache after insertion");
            }
            {
                std::vector<size_t> vc_shape_vec = {value_cache_shape[0],
                                                    value_cache_shape[1],
                                                    value_cache_shape[2],
                                                    value_cache_shape[3]};
                debug_print_tensor(value_cache, vc_shape_vec, "Value Cache after insertion");
            }
        }

        // Per-head attention
        for (size_t head = 0; head < ctx.num_heads; ++head) {
            std::cout << "  >>> Head " << head << " <<<\n\n";

            // 3.2.1) Extract query vector for this token and head
            const T* q_vector = query + token_idx * ctx.query_feature_size + head * ctx.query_head_size;
            {
                std::vector<size_t> qv_shape = {ctx.query_head_size};
                debug_print_tensor(
                    q_vector,
                    qv_shape,
                    "    q_vector [token=" + std::to_string(token_idx) + ", head=" + std::to_string(head) + "]");
            }

            // Determine past_tokens, new_tokens, total_keys
            int32_t past_tokens = past_lens ? past_lens[seq_idx] : 0;
            int32_t new_tokens =
                subsequence_begins ? (subsequence_begins[seq_idx + 1] - subsequence_begins[seq_idx]) : 0;
            int32_t total_keys = std::min(past_tokens + new_tokens, ctx.max_context_length);

            debug_print_int_scalar(past_tokens, "    past_tokens");
            debug_print_int_scalar(new_tokens, "    new_tokens");
            debug_print_int_scalar(total_keys, "    total_keys");

            // 3.2.2) Allocate and compute raw scores vector
            std::vector<T> scores(total_keys, T(0));
            for (int32_t k = 0; k < total_keys; ++k) {
                std::cout << "    -- Computing score for k = " << k << " --\n\n";

                T score_value = T(0);
                if (k < past_tokens) {
                    int32_t block_id, offset;
                    if (find_cached_token(seq_idx, k, ctx, block_id, offset)) {
                        std::cout << "      Found cached token at block_id = " << block_id << ", offset = " << offset
                                  << "\n\n";

                        if (block_id == ctx.block_indices[ctx.block_indices_begins[seq_idx]] &&
                            offset < ctx.sliding_window) {
                            std::cout << "      Token is within sliding window cutoff. Assigning -inf.\n\n";
                            scores[k] = -std::numeric_limits<T>::infinity();
                            continue;
                        } else {
                            std::cout << "      Computing score via cached key.\n\n";
                            score_value = compute_score_for_cached_key(q_vector, head, ctx, block_id, offset);
                        }
                    }
                } else {
                    int32_t new_idx =
                        subsequence_begins ? (subsequence_begins[seq_idx] + (k - past_tokens)) : (k - past_tokens);
                    std::cout << "      Computing score via new key at token index = " << new_idx << "\n\n";
                    score_value = compute_score_for_new_key(q_vector, head, new_idx, ctx);
                }

                T alibi = alibi_slopes ? ((T*)alibi_slopes)[head] : T(0);
                std::cout << "      Raw dot-product score = " << std::fixed << std::setprecision(6) << score_value
                          << "\n";
                std::cout << "      Alibi slope for head = " << alibi << "\n";
                T biased_score = score_value * scale + alibi * T(-(total_keys - k - 1));
                std::cout << "      Biased (scaled + alibi) score = " << biased_score << "\n\n";

                scores[k] = biased_score;
            }

            // Raw scores array
            {
                std::vector<size_t> scores_shape = {(size_t)total_keys};
                debug_print_tensor(scores.data(),
                                   scores_shape,
                                   "    raw_scores before softmax [token=" + std::to_string(token_idx) +
                                       ", head=" + std::to_string(head) + "]");
            }

            // Softmax normalization
            softmax(scores);
            {
                std::vector<size_t> scores_shape = {(size_t)total_keys};
                debug_print_tensor(scores.data(),
                                   scores_shape,
                                   "    normalized_scores after softmax [token=" + std::to_string(token_idx) +
                                       ", head=" + std::to_string(head) + "]");
            }

            // Accumulate values into output_vector
            std::vector<T> output_vector(ctx.value_head_size, T(0));
            for (int32_t k = 0; k < total_keys; ++k) {
                std::cout << "    -- Accumulating value for k = " << k << " --\n\n";
                T weight = scores[k];
                std::cout << "      weight = " << std::fixed << std::setprecision(6) << weight << "\n\n";

                if (k < past_tokens) {
                    int32_t block_id, offset;
                    if (find_cached_token(seq_idx, k, ctx, block_id, offset)) {
                        if (!(block_id == ctx.block_indices[ctx.block_indices_begins[seq_idx]] &&
                              offset < ctx.sliding_window)) {
                            std::cout << "      Using cached value for accumulation.\n";
                            accumulate_value_for_cached_key(head, ctx, block_id, offset, weight, output_vector);
                        } else {
                            std::cout << "      Skipping accumulation (within sliding window cutoff).\n";
                        }
                    }
                } else {
                    int32_t new_idx =
                        subsequence_begins ? (subsequence_begins[seq_idx] + (k - past_tokens)) : (k - past_tokens);
                    std::cout << "      Using new value at token index = " << new_idx << " for accumulation.\n";
                    accumulate_value_for_new_key<T>(new_idx, head, ctx, weight, output_vector);
                }

                // Partial output_vector after each accumulation
                {
                    std::vector<size_t> outv_shape = {ctx.value_head_size};
                    debug_print_tensor(output_vector.data(),
                                       outv_shape,
                                       "      partial_output_vector [token=" + std::to_string(token_idx) +
                                           ", head=" + std::to_string(head) + ", after k=" + std::to_string(k) + "]");
                }

                size_t global_index = seq_idx * ctx.max_context_length + k;
                score[global_index] = scores[k];
            }

            // Final output_vector for this head
            {
                std::vector<size_t> outv_shape = {ctx.value_head_size};
                debug_print_tensor(output_vector.data(),
                                   outv_shape,
                                   "    final_output_vector [token=" + std::to_string(token_idx) +
                                       ", head=" + std::to_string(head) + "]");
            }

            // Copy output_vector into the out buffer
            T* destination = out + token_idx * ctx.value_feature_size + head * ctx.value_head_size;
            std::memcpy(destination, output_vector.data(), ctx.value_head_size * sizeof(T));

            std::cout << "\n  <<< End of head " << head << " >>>\n\n";
        }

        std::cout << "----- End processing token " << token_idx << " -----\n\n";
    }
    std::cout << "===== End of paged_attention =====\n\n";
}

}  // namespace ov::reference
