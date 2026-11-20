// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Lowers ov::op::internal::PagedSparseAttention to the GPU SparsePA op and lays out its key / value caches.
///
/// The head geometry SparsePA is specialized for comes from the `num_k_heads` / `k_head_size` / `num_v_heads` /
/// `v_head_size` runtime info the paged-attention conversion stamps on the op (as for PagedAttentionExtension), or else
/// from static cache and projection shapes: num_kv_heads is key_cache dim 1, the head sizes are the key / value widths
/// over it and num_heads is the query width over the key head size.
///
/// When the caches are model Parameters, this pass owns their layout (ConvertPagedAttnInputs must skip
/// PagedSparseAttention: the PagedAttention page layout is not the one SparsePA reads). Caches declared with a static
/// page dimension keep it, in f16; the others (as the paged-attention conversion emits them) use the configured page size:
///  - f16 caches are [blocks, num_kv_heads, page_size, head_size];
///  - with an 8- or 4-bit cache precision they are u8 SparsePA pages [blocks, num_kv_heads, page_size + parameter rows,
///    bytes per row] (see sparse_kv_quant.hpp), the key per channel or per token as configured, the value per token.
///    A geometry SparsePA cannot quantize (e.g. a head size that is not a power of two of at least 32) keeps f16 caches.
/// Caches that are not Parameters must already be f16 [blocks, num_kv_heads, page_size, head_size].
/// PagedSparseAttention that cannot be expressed as SparsePA is left as is.
class ConvertPagedSparseAttentionToSparsePA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertPagedSparseAttentionToSparsePA");

    struct CacheConfig {
        ov::element::Type precision = ov::element::f16;  ///< KV cache precision; 8- or 4-bit types quantize the caches
        bool key_by_channel = true;                      ///< key quantized per channel (else per token)
        size_t page_size = 16;                           ///< tokens per page of Parameter caches
    };

    explicit ConvertPagedSparseAttentionToSparsePA(const CacheConfig& cache_config);
};

}  // namespace ov::intel_gpu
