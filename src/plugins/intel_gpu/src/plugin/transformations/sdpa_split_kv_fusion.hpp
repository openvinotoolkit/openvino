// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Fuses the fused v13::ScaledDotProductAttention that ov::pass::SDPASplitAttentionFusionMatcher
/// builds for Gemma-style models keeping the persistent KV cache and the current step's KV separate,
/// into a single split-KV ov::intel_gpu::op::SDPA (split_kv = true), WITHOUT materializing a
/// full-cache KV Concat.
///
/// ov::pass::SDPASplitAttentionFusionMatcher is registered immediately before this pass in the GPU
/// pipeline (see transformations_pipeline.cpp) and has already rewritten the raw
/// MatMul/Concat/Softmax/VariadicSplit sub-graph into:
///
///     sdpa = v13::SDPA(Q, Concat(K_cache, K_new, seq_axis), Concat(V_cache, V_new, seq_axis), mask, scale=1.0)
///
/// On GPU that Concat copies the entire cache every step (its first input is the host-owned
/// kv_cache Parameter, which cannot be concatenated in-place). This pass instead un-does the K/V
/// concat and builds a split-KV op that attends over the logical concatenation directly, so no
/// per-step cache copy is created.
///
/// Fires for DECODE ONLY (q_len == 1; K_new / V_new contribute the single newest step). The forked
/// split-KV kernel mirrors sdpa_opt's single-token decode path (KV-sequence partitioning +
/// finalization) with the cache loops kept byte-identical to sdpa_opt and the new chunk appended as
/// a tail on the last partition. Prefill / multi-token chunks (q_len > 1) fall through to the
/// regular multi-token SDPA path. Scope: Q/K/V must be static 4D, the new-chunk seq length must be
/// static, and KV-cache compression / indirect / sink are not supported. Otherwise it does not fire.
class SDPASplitKVFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPASplitKVFusion");
    SDPASplitKVFusion();
};

}  // namespace ov::intel_gpu
