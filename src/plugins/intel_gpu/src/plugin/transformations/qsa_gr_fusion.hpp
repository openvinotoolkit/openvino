// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Recognizes the Gated Residual READ / MIX (and FINAL_MIX) pattern and replaces
/// it with ov::intel_gpu::op::GatedResidual.
///
/// Canonical sub-graph (all activations are token-major [T, C*D]):
///   r3    = Reshape(R,   [T, C, D])
///   n3    = RMS(r3, gamma, eps)                 // branch-wise grouped RMSNorm, width D
///   rbar  = Reshape(n3,  [T, C*D])
///   low   = MatMul(rbar, W_down, transpose_b)   // W_down: [L, C*D]
///   act   = Swish(low * 1/C)
///   up    = MatMul(act,  W_up,   transpose_b)   // W_up:   [C*D, L]
///   wm    = Sigmoid(up) * rbar
///   mixed = ReduceSum(Reshape(wm, [T, C, D]), axis=1) * 1/C
///
/// The pass refuses to fuse (and leaves the generic graph untouched) unless it can prove:
///   * C is static and equal to 4 (first supported branch count);
///   * the grouped-norm width is exactly D, i.e. the norm is per branch, not over C*D;
///   * the reduce axis is the branch axis;
///   * both 1/C scalings are present;
///   * no other consumer would observe a semantic change.
class GatedResidualReadFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedResidualReadFusion");
    GatedResidualReadFusion();
};

/// @brief Recognizes the Gated Residual WRITE / COMBINE pattern.
///
/// Canonical sub-graph:
///   inj  = MatMul(rbar, W_inject, transpose_b)  // W_inject: [C, C*D]
///   S    = 2 * Sigmoid(inj * 1/C)               // per-branch scalar gate in (0, 2)
///   R'   = Reshape(Reshape(R, [T,C,D]) + Reshape(S,[T,C,1]) * Reshape(Y,[T,1,D]), [T, C*D])
///
/// The `2 *` factor must be present: the write gate range is (0, 2), not (0, 1).
class GatedResidualWriteFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedResidualWriteFusion");
    GatedResidualWriteFusion();
};

/// @brief Recognizes the QSA indexer signature and replaces the indexer + sparse
/// attention sub-graph with ov::intel_gpu::op::QSA.
///
/// The signature that identifies a QSA layer is the indexer score:
///   scores = ReduceSum(ReLU(MatMul(q_idx, k_block, transpose_b)), axis=head) * 1/sqrt(Di)
///   topk   = TopK(scores, block_topk)
/// The ReLU must sit BEFORE the head reduction and the 1/sqrt(Di) AFTER it; any other
/// order is a different operator and must not be fused.
///
/// The pass bails out - keeping the original generic graph - whenever a required tensor
/// (indexer projection weights, RoPE table, caches, page table, ...) cannot be proven,
/// or when the attribute constraints of ov::intel_gpu::op::QSA::validate_config() fail.
class QSAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSAFusion");
    QSAFusion();
};

/// @brief Recognizes the SDPA-style (dense, KVCache/ReadValue-managed) QSA layer and
/// replaces the indexer + main-attention sub-graph with
/// `ov::intel_gpu::op::QSAIndexerCache` + `ov::intel_gpu::op::QSASDPA`.
///
/// This is the "no physical page table" sibling of `QSAFusion`: instead of a paged main
/// KV cache, the main attention is an ordinary `ov::op::v13::ScaledDotProductAttention`
/// fed by a dense, per-sequence-growing `ReadValue`/`Concat` (or already-fused
/// `ov::intel_gpu::op::KVCache`) state, and the indexer's own raw-K history is a
/// *separate* `ReadValue`/`Concat` state (there is normally no separate block-summary
/// state at all -- the decomposed graph recomputes it from the full raw-K history on
/// every call via a mean-pooling reduction, which is exactly the O(total_len) redundant
/// work this fusion removes).
///
/// The pass reuses `QSAFusion`'s indexer-score signature detection (same ReLU ->
/// head-reduction -> 1/sqrt(Di) -> [mask] -> TopK proof) and additionally requires, all
/// as structural / shape facts (no literal weight-name matching):
///   * the indexer Q and K legs trace back to the *same* projection MatMul (the fused
///     `[Hidx*Di + Di, D]` `INDEXER_QK_WEIGHT` convention);
///   * the indexer K leg has its own `ReadValue`-backed history, distinct from the main
///     attention's;
///   * a `ScaledDotProductAttention` node is reachable downstream of the indexer score;
///   * its K/V operands each trace back to a `ReadValue`-backed Variable (distinct from
///     the indexer's), with a fully **static** declared shape and batch size 1 -- `v1`
///     of this pass only supports the single-sequence, static-capacity case (matching
///     `QSASDPA`'s own compile-time `pa_block_size` requirement);
///   * a single already rotary-packed `[max_positions, indexer_rotary_dim]` constant
///     feeds the indexer's own RoPE application (a model that only exposes separate,
///     un-interleaved cos/sin tensors -- as `full_cos`/`full_sin` Parameters, the
///     convention seen in real HF-style exports -- cannot be proven safe to repack here
///     and is left untouched; repacking such tables is a documented future increment,
///     not attempted by `v1`).
///
/// Any of these that cannot be proven causes the pass to bail out (leaving the original,
/// generic graph untouched, mirroring `QSAFusion`'s own conservative philosophy) rather
/// than guess. `q_proj`/`k_proj`/`v_proj`/`o_proj` and any output-gate logic are left
/// completely untouched either way: `QSASDPA`, like `QSA`, only owns the indexer +
/// sparse-attention math, not the surrounding projections.
class QSASDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSASDPAFusion");
    QSASDPAFusion();
};

/// @brief Convenience container running all QSA / GR pattern transformations.
class QSAGatedResidualFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("QSAGatedResidualFusion");
    QSAGatedResidualFusion() {
        add_matcher<GatedResidualReadFusion>();
        add_matcher<GatedResidualWriteFusion>();
        add_matcher<QSAFusion>();
        add_matcher<QSASDPAFusion>();
    }
};

}  // namespace ov::intel_gpu
