// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {

// Replaces the query-aware sparse attention (opset17 QSAPrepare -> SparseAttentionIndexer -> SparseAttention)
// of a stateless LLM with standard operations, the attention itself becoming a dense
// ScaledDotProductAttention over the whole KV cache:
//
// 1. QSAPrepare and SparseAttentionIndexer are decomposed, so the block selection is kept unchanged.
// 2. SparseAttention becomes a dense SDPA whose boolean mask allows exactly the keys SparseAttention
//    attends to: the tokens of the selected key blocks and the causal tail of the incomplete block of
//    every query. The result is the same as of the sparse computation for any KV length.
// 3. The logical position of every key slot is rebuilt from attention_mask as CumSum(mask) - 1, which
//    covers the NPUW [past][unused][current] KV layout and left-padded prefill chunks (the operations
//    assume right-aligned valid keys). A migration to right padding must revisit query_positions and
//    the QSAPrepare offset, which place the queries on the last L key slots and the valid new tokens
//    on the last rows of the chunk.
// 4. The number of key blocks, the only data-dependent dimension, becomes the constant max_kv_len / r,
//    and the QSA states (block_index_key, incomplete_index_key, incomplete_index_rope) are renamed to
//    cache_params.{past|present}.<kind>.N, so the pipeline handles them like the linear-attention
//    states: static shapes and a full-tensor present -> past copy.
//
// Run once on the stateless model, before the prefill and generate models are cloned.
class ConvertSparseAttentionToDense : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ConvertSparseAttentionToDense");

    explicit ConvertSparseAttentionToDense(uint32_t max_kv_len) : m_max_kv_len(max_kv_len) {}

    // Returns true when the model had sparse attention operations and they have been replaced.
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    uint32_t m_max_kv_len;
};

}  // namespace npuw
}  // namespace ov
