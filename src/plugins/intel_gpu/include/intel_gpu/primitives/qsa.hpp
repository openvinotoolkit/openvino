// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/op/qsa.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief qsa (Qwen Sparse Attention) primitive.
/// @details Owns four CM kernel families: main KV cache update, indexer prepare,
/// block score/top-k and selected-block sparse attention. See ov::intel_gpu::op::QSA.
struct qsa : public primitive_base<qsa> {
    CLDNN_DECLARE_PRIMITIVE(qsa)

    using QSAInputIdx = ov::intel_gpu::op::QSA::InputIdx;

    qsa() : primitive_base("", {}) {}

    // skip_kv_write: when true, the Q0 main-KV-cache-write stage (qsa_kv_cache_update)
    // is never dispatched -- used by the QSASDPA dense/degenerate-paging lowering, where
    // an external ov::intel_gpu::op::KVCache node has already appended this call's new
    // K/V into the (degenerate-paged) cache before this primitive runs, so Q0's write
    // would be redundant. Every existing caller (the paged QSA op) leaves this at its
    // default `false`, so existing behavior is completely unchanged.
    //
    // prepare_only: when true, ONLY Q1 (qsa_prepare / qsa_project_dpas) is dispatched --
    // Q0 is skipped exactly as if skip_kv_write were also set, and Q2 (score/top-k) and
    // Q3 (sparse attention) are skipped too. This is the primitive-level building block
    // for the future QSAIndexerCache op: Q1 is reused
    // completely unchanged, just addressed through a degenerate (one-page-per-sequence)
    // page table, to own the indexer raw-K/summary persistent state independently of the
    // main-attention side. The primitive's declared attention output is left uninitialized
    // in this mode; callers only care about the mutated raw-K/summary cache buffers.
    qsa(const primitive_id& id,
        const std::vector<input_info>& inputs,
        const ov::intel_gpu::op::QSA::Config& config,
        bool skip_kv_write = false,
        bool prepare_only = false)
        : primitive_base(id, inputs),
          heads_num(config.heads_num),
          kv_heads_num(config.kv_heads_num),
          k_head_size(config.k_head_size),
          v_head_size(config.v_head_size),
          hidden_size(config.hidden_size),
          indexer_heads(config.indexer_heads),
          indexer_kv_heads(config.indexer_kv_heads),
          indexer_head_dim(config.indexer_head_dim),
          indexer_rotary_dim(config.indexer_rotary_dim),
          budget(config.budget),
          compress_ratio(config.compress_ratio),
          qsa_block_tokens(config.qsa_block_tokens),
          pa_block_size(config.pa_block_size),
          block_topk(config.block_topk),
          scale(config.scale),
          indexer_norm_eps(config.indexer_norm_eps),
          has_output_gate(config.has_output_gate),
          is_causal(config.is_causal),
          skip_kv_write(skip_kv_write),
          prepare_only(prepare_only) {
        // prepare_only declares 2 outputs (mutated raw-K cache, mutated summary cache --
        // see qsa_inst::calc_output_layouts) instead of the normal single attention
        // output; every existing (non-prepare_only) caller is unaffected since this only
        // changes num_outputs away from its primitive_base default of 1 when
        // prepare_only is set.
        if (prepare_only) {
            num_outputs = 2;
        }
    }

    size_t heads_num = 0;
    size_t kv_heads_num = 0;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t hidden_size = 0;
    size_t indexer_heads = 0;
    size_t indexer_kv_heads = 1;
    size_t indexer_head_dim = 0;
    size_t indexer_rotary_dim = 0;
    size_t budget = 0;
    size_t compress_ratio = 0;
    size_t qsa_block_tokens = 0;
    size_t pa_block_size = 0;
    size_t block_topk = 0;
    float scale = 0.0f;
    float indexer_norm_eps = 1e-6f;
    bool has_output_gate = false;
    bool is_causal = true;
    bool skip_kv_write = false;
    bool prepare_only = false;

    /// @brief Pre-allocated width of the selected-token metadata: budget + r - 1.
    size_t max_selected_tokens() const {
        return budget + compress_ratio - 1;
    }

    ov::intel_gpu::op::QSA::Config get_config() const {
        ov::intel_gpu::op::QSA::Config config;
        config.heads_num = heads_num;
        config.kv_heads_num = kv_heads_num;
        config.k_head_size = k_head_size;
        config.v_head_size = v_head_size;
        config.hidden_size = hidden_size;
        config.indexer_heads = indexer_heads;
        config.indexer_kv_heads = indexer_kv_heads;
        config.indexer_head_dim = indexer_head_dim;
        config.indexer_rotary_dim = indexer_rotary_dim;
        config.budget = budget;
        config.compress_ratio = compress_ratio;
        config.qsa_block_tokens = qsa_block_tokens;
        config.pa_block_size = pa_block_size;
        config.block_topk = block_topk;
        config.scale = scale;
        config.indexer_norm_eps = indexer_norm_eps;
        config.has_output_gate = has_output_gate;
        config.is_causal = is_causal;
        return config;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, heads_num);
        seed = hash_combine(seed, kv_heads_num);
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, hidden_size);
        seed = hash_combine(seed, indexer_heads);
        seed = hash_combine(seed, indexer_kv_heads);
        seed = hash_combine(seed, indexer_head_dim);
        seed = hash_combine(seed, indexer_rotary_dim);
        seed = hash_combine(seed, budget);
        seed = hash_combine(seed, compress_ratio);
        seed = hash_combine(seed, qsa_block_tokens);
        seed = hash_combine(seed, pa_block_size);
        seed = hash_combine(seed, block_topk);
        seed = hash_combine(seed, scale);
        seed = hash_combine(seed, indexer_norm_eps);
        seed = hash_combine(seed, has_output_gate);
        seed = hash_combine(seed, is_causal);
        seed = hash_combine(seed, skip_kv_write);
        seed = hash_combine(seed, prepare_only);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto r = downcast<const qsa>(rhs);
        return heads_num == r.heads_num && kv_heads_num == r.kv_heads_num && k_head_size == r.k_head_size && v_head_size == r.v_head_size &&
               hidden_size == r.hidden_size && indexer_heads == r.indexer_heads && indexer_kv_heads == r.indexer_kv_heads &&
               indexer_head_dim == r.indexer_head_dim && indexer_rotary_dim == r.indexer_rotary_dim && budget == r.budget &&
               compress_ratio == r.compress_ratio && qsa_block_tokens == r.qsa_block_tokens && pa_block_size == r.pa_block_size && block_topk == r.block_topk &&
               scale == r.scale && indexer_norm_eps == r.indexer_norm_eps && has_output_gate == r.has_output_gate && is_causal == r.is_causal &&
               skip_kv_write == r.skip_kv_write && prepare_only == r.prepare_only;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<qsa>::save(ob);
        ob << heads_num;
        ob << kv_heads_num;
        ob << k_head_size;
        ob << v_head_size;
        ob << hidden_size;
        ob << indexer_heads;
        ob << indexer_kv_heads;
        ob << indexer_head_dim;
        ob << indexer_rotary_dim;
        ob << budget;
        ob << compress_ratio;
        ob << qsa_block_tokens;
        ob << pa_block_size;
        ob << block_topk;
        ob << scale;
        ob << indexer_norm_eps;
        ob << has_output_gate;
        ob << is_causal;
        ob << skip_kv_write;
        ob << prepare_only;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<qsa>::load(ib);
        ib >> heads_num;
        ib >> kv_heads_num;
        ib >> k_head_size;
        ib >> v_head_size;
        ib >> hidden_size;
        ib >> indexer_heads;
        ib >> indexer_kv_heads;
        ib >> indexer_head_dim;
        ib >> indexer_rotary_dim;
        ib >> budget;
        ib >> compress_ratio;
        ib >> qsa_block_tokens;
        ib >> pa_block_size;
        ib >> block_topk;
        ib >> scale;
        ib >> indexer_norm_eps;
        ib >> has_output_gate;
        ib >> is_causal;
        ib >> skip_kv_write;
        ib >> prepare_only;
    }
};

}  // namespace cldnn
