// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/// rt_info key on a model input whose tensor the device mutates in place. Such inputs
/// must be bound to a same-context RemoteTensor/USM host tensor of the port's exact
/// element type; a host copy would silently drop the update.
inline constexpr const char* in_place_cache_rt_key = "GPU_IN_PLACE_CACHE";

/// @brief Recognizes a decomposed, non-paged (dense) Qwen Sparse Attention (QSA) layer
/// and replaces its indexer + attention sub-graph with
/// `ov::op::internal::QSAIndexer` + `ov::op::internal::SparseSDPA`.
///
/// The signature that identifies a QSA indexer is the same block score used by the
/// legacy fused-op detector: `scores = ReduceSum(ReLU(MatMul(q_idx, k_block,
/// transpose_b)), axis=head) * 1/sqrt(Di)` followed by `TopK(scores, block_topk)`.
///
/// The indexer state (`summary_cache`, `past_incomplete_raw_k`, `past_incomplete_rope`)
/// follows QSAIndexer's functional state interface: the pass creates one f16 `Variable`
/// per state tensor, reads it with a zero-initialized `ReadValue`, grows `summary_cache`
/// on its sequence axis to `floor(key_length / r)` entries, and feeds the updated state
/// outputs back with `Assign`. The decomposed source pattern must already keep the
/// indexer's raw-key history in a `ReadValue`-backed `Variable` (what distinguishes a real
/// incremental graph from a one-shot reference computation); that history and the mean-pool
/// recomputation it feeds are retired. `key_length` is the main K length after the call
/// (shared by QSAIndexer and SparseSDPA) and `rope_cos`/`rope_sin` are the full-width rows
/// the source graph gathers from its RoPE table for each token. Only causal,
/// default-scale attention (or a mask derived from the indexer's own TopK) is replaced.
/// The pass bails out -- leaving the original generic graph untouched -- whenever any
/// required fact cannot be proven.
class DenseSparseAttentionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DenseSparseAttentionFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/// @brief Recognizes a decomposed, paged (continuous-batching) QSA layer and replaces
/// its indexer + attention sub-graph with `ov::op::internal::PagedQSAIndexer` +
/// `ov::op::internal::SparsePA`.
///
/// Reuses `DenseSparseAttentionFusion`'s indexer-score signature detection, but requires
/// a reachable `ov::op::PagedAttentionExtension` downstream of the indexer's TopK output
/// instead of a dense `ScaledDotProductAttention`. `PagedAttentionExtension` already
/// carries the page table (`block_indices`/`block_indices_begins`/`subsequence_begins`/
/// `past_lens`) as plain inputs -- continuous-batching graphs never use `ReadValue` for
/// this -- so those four tensors are reused verbatim for `PagedQSAIndexer`/`SparsePA`
/// without being recreated. The indexer's own `summary_cache`/`indexer_raw_k_cache`/
/// `indexer_rope_cache` are, like the dense case, modeled as brand-new tagged
/// `Parameter`s appended to the model (page geometry from `key_cache`), and
/// `rope_cos`/`rope_sin` are the RoPE rows the source graph gathers for each token.
/// Optional `PagedAttentionExtension` features (sliding window, alibi, sinks, xattention,
/// cache rotation, adaptive RKV, qq_bias, score/diversity outputs) and integer KV caches
/// must be provably disabled, otherwise the pass bails out.
///
/// Unlike `DenseSparseAttentionFusion`, this path has no validated worked reference in
/// this codebase (the legacy `QSAFusion` it supersedes never actually replaced a paged
/// graph, only tagged `rt_info`) -- it is written to the same conservative,
/// prove-or-bail-out discipline, but should be treated as a lower-confidence, untested
/// structural design until checked against a real paged QSA export.
class PagedSparseAttentionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PagedSparseAttentionFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/// @brief Convenience container running both sparse-attention pattern rewrites.
class SparseAttentionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SparseAttentionFusion");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        bool changed = false;
        changed |= DenseSparseAttentionFusion().run_on_model(model);
        changed |= PagedSparseAttentionFusion().run_on_model(model);
        return changed;
    }
};

}  // namespace ov::intel_gpu
