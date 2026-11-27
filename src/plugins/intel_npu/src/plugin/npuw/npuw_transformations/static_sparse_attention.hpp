// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {

// Makes the decomposed sparse attention (QSA: QSAPrepare -> SparseAttentionIndexer -> SparseAttention)
// of a stateless model shape-static, keeping the sparse block selection:
//
// 1. num_blocks = ReduceMax(key_length / r) is the only data-dependent dimension: it sizes the
//    block_index_key state. It is replaced by the constant max_kv_len / r, so the state becomes
//    [batch, max_kv_len / r, D] in every model variant. The blocks beyond the actual key_length are
//    zero and stay invisible to the indexer and to SparseAttention, so the selection is unchanged.
// 2. The key slot -> sequence position mapping Range(S) - (S - key_length) assumes the valid keys
//    are right-aligned. It is rebuilt from attention_mask as CumSum(mask) - 1 (-1 for masked slots),
//    which also matches the NPUW [past][unused][current] KV layout.
// 3. The QSA states (block_index_key, incomplete_index_key, incomplete_index_rope) are renamed to
//    cache_params.{past|present}.<kind>.N, so the pipeline handles them like the linear-attention
//    states: static shapes and a full-tensor present -> past copy.
//
// Run once on the stateless model, before the prefill and generate models are cloned.
class MakeSparseAttentionStatic : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::MakeSparseAttentionStatic");

    explicit MakeSparseAttentionStatic(uint32_t max_kv_len) : m_max_kv_len(max_kv_len) {}

    // Returns true when the model has QSA states and they have been made static.
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    uint32_t m_max_kv_len;
};

}  // namespace npuw
}  // namespace ov
