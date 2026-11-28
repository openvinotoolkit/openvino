// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "intel_gpu/op/sparse_kv_quant.hpp"

namespace ov::intel_gpu::op {

/// \brief Positional inputs of ov::op::internal::PagedSparseAttention, as the GPU plugin reads them.
struct SparsePAInputIdx {
    static constexpr size_t QUERY = 0;                   ///< [tokens, num_heads*k_head_size]
    static constexpr size_t KEY = 1;                     ///< [tokens, num_kv_heads*k_head_size], current step
    static constexpr size_t VALUE = 2;                   ///< [tokens, num_kv_heads*v_head_size], current step
    static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, num_kv_heads, page_size, k_head_size], writable (u8 pages when quantized)
    static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, num_kv_heads, page_size, v_head_size], writable (u8 pages when quantized)
    static constexpr size_t PAST_LENS = 5;               ///< [num_sequences]
    static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_sequences+1]
    static constexpr size_t BLOCK_INDICES = 7;           ///< [total_logical_blocks]
    static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_sequences+1]
    static constexpr size_t SELECTED_BLOCK_INDICES = 9;  ///< [tokens, num_selection_heads, K_max]
    static constexpr size_t SELECTED_BLOCK_COUNT = 10;   ///< [tokens, num_selection_heads]
    static constexpr size_t SCALE = 11;                  ///< scalar
    static constexpr size_t SINKS = 12;                  ///< [num_heads], or an empty tensor (no sinks)

    static constexpr size_t COUNT = 13;
};

/// \brief Head geometry and cache layout of the GPU paged sparse attention (cldnn::sparse_pa).
///
/// ov::op::internal::PagedSparseAttention is lowered to cldnn::sparse_pa directly by CreatePagedSparseAttentionOp, as
/// PagedAttentionExtension is lowered to cldnn::paged_attention: the head geometry comes from the `num_k_heads` /
/// `k_head_size` / `num_v_heads` / `v_head_size` runtime info (or the input shapes), the cache quantization from the
/// cache element type and the KV cache options. The caches are laid out by ConvertPagedSparseAttnInputs.
/// This is not an op: it is the configuration shared by the lowering and the primitive.
struct SparsePAConfig {
    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t key_block_size = 0;    ///< must divide the page size
    size_t query_block_size = 1;  ///< only 1 is implemented
    bool causal = true;
    size_t page_size = 0;   ///< tokens per page; 0 = taken from key_cache dim 2 (f16 caches only)
    KVQuant key_quant{};    ///< quantization of the key cache
    KVQuant value_quant{};  ///< quantization of the value cache

    bool kv_quantized() const {
        return key_quant.enabled() || value_quant.enabled();
    }

    /// Whether the kernels support this configuration, else false with the `reason`.
    bool validate(std::string& reason) const;
};

}  // namespace ov::intel_gpu::op
