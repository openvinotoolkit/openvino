// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// rt_info key on a model input whose tensor the device mutates in place. Such inputs
/// must be bound to a same-context RemoteTensor/USM host tensor of the port's exact
/// element type; a host copy would silently drop the update.
inline constexpr const char* in_place_cache_rt_key = "GPU_IN_PLACE_CACHE";

/// @brief Fuses the public opset17 `QSAPrepare` + `SparseAttentionIndexer` pair into
/// `ov::intel_gpu::op::QSAIndexer`.
///
/// The three states of `QSAPrepare` (block index keys, raw keys and RoPE row of the incomplete
/// block) keep flowing through the model's own ReadValue/Assign pairs; the fused op returns their
/// new values. The block-key state is padded with zero rows up to the number of complete blocks
/// of the longest sequence, which the kernels fill in. The pass leaves the ops untouched whenever
/// the GPU kernels cannot run them (`index_weights`, several selection heads, missing RoPE or
/// norm weight, non-f16 data).
class QSAIndexerFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSAIndexerFusion");
    QSAIndexerFusion();
};

/// @brief Maps the public opset17 `SparseAttention` to `ov::intel_gpu::op::SparseSDPA`.
/// Attention sinks, non-causal attention and query blocks are not supported by the kernels.
class SparseSDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SparseSDPAFusion");
    SparseSDPAFusion();
};

/// @brief Maps `ov::op::internal::PagedQSAIndexer` to `ov::intel_gpu::op::PagedQSAIndexer` and
/// marks its three caches, which the device updates in place.
class PagedQSAIndexerConversion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedQSAIndexerConversion");
    PagedQSAIndexerConversion();
};

/// @brief Maps `ov::op::internal::PagedSparseAttention` to `ov::intel_gpu::op::SparsePA`, which
/// carries the head geometry the kernels are specialized for.
class PagedSparseAttentionConversion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSparseAttentionConversion");
    PagedSparseAttentionConversion();
};

/// @brief All Qwen Sparse Attention conversions of the public ops to GPU plugin ops.
class SparseAttentionFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("SparseAttentionFusion");
    SparseAttentionFusion();
};

}  // namespace ov::intel_gpu
