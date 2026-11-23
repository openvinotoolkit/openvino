// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

namespace ov::intel_gpu {

/// Lays out the key / value caches of ov::op::internal::PagedSparseAttention for cldnn::sparse_pa.
///
/// This is ov::pass::ConvertPagedAttnInputs (the pass that lays out the PagedAttentionExtension caches) with the GPU
/// sparse attention cache layout, see sparse_kv_quant.hpp: [blocks, num_kv_heads, page_size, head_size] f16 caches,
/// or, with an 8- or 4-bit KV cache precision, u8 pages [blocks, num_kv_heads, page_size + parameter rows, bytes per
/// row] with the key quantized per channel or per token as configured and the value per token.
///
/// A distinct pass type gives it its own transformation callback: register it with `set_callback(skip)` (see skip())
/// so that it only handles PagedSparseAttention, while the ConvertPagedAttnInputs instance of PagedAttentionExtension
/// skips PagedSparseAttention. The common pass, and so the CPU plugin, keeps its own layout.
class ConvertPagedSparseAttnInputs : public ov::pass::ConvertPagedAttnInputs {
public:
    OPENVINO_RTTI("ConvertPagedSparseAttnInputs", "0", ov::pass::ConvertPagedAttnInputs);

    struct CacheConfig {
        ov::element::Type precision = ov::element::f16;  ///< KV cache precision; 8- or 4-bit types quantize the caches
        bool key_by_channel = true;                      ///< key quantized per channel (else per token)
        size_t page_size = 16;                           ///< tokens per page
    };

    explicit ConvertPagedSparseAttnInputs(const CacheConfig& cache_config);

    /// The transformation callback of this pass: true (skip) for any node that is not a PagedSparseAttention with the
    /// head geometry runtime info and key / value caches that are Parameters it alone reads.
    static bool skip(const std::shared_ptr<const ov::Node>& node);

    /// The cache precision for the PagedSparseAttention layers of `model`: `precision`, or f16 when a layer has a
    /// geometry the quantized SparsePA caches do not support (e.g. a head size that is not a power of two of at least
    /// 32), with a warning.
    static ov::element::Type select_precision(const ov::Model& model, const CacheConfig& cache_config);
};

}  // namespace ov::intel_gpu
