// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Fuses the "split attention" sub-graph into a single split-KV
/// ov::intel_gpu::op::SDPA (split_kv = true), WITHOUT materializing a full-cache KV Concat.
///
/// The matched sub-graph (produced by Gemma-style models that keep the persistent KV cache and
/// the current step's KV separately) is:
///
///     qk_cache = MatMul(Q, K_cache)
///     qk_new   = MatMul(Q, K_new)
///     scores   = Concat(qk_cache, qk_new, axis=-1)
///     masked   = Add(scores, mask)
///     probs    = Softmax(masked, axis=-1)
///     [pc, pn] = VariadicSplit(probs, axis=-1)
///     attn_c   = MatMul(pc, V_cache)
///     attn_n   = MatMul(pn, V_new)
///     output   = Add(attn_c, attn_n)
///
/// A naive fusion would feed a single v13::SDPA with Concat(K_cache, K_new) / Concat(V_cache,
/// V_new). On GPU that concat copies the entire cache every step (its first input is the host-owned
/// kv_cache Parameter, which cannot be concatenated in-place). This pass instead builds a split-KV
/// op that attends over the logical concatenation directly, so no per-step cache copy is created.
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
