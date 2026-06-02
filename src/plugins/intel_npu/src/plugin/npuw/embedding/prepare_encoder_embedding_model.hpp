// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

// Prepares a non-autoregressive bidirectional encoder (e.g. BERT) text-embedding model for the
// NPUW embedding path. Unlike PrepareTextEmbeddingModel (autoregressive / Qwen3-Embedding-style),
// it does NOT inject KV-cache parameters, RoPE position ids, or a causal attention mask: an
// encoder is self-contained (builds its own bidirectional mask from `attention_mask`, uses
// learned absolute positions, has no KV cache) and processes the whole sequence in one forward.
class PrepareEncoderEmbeddingModel : public ov::pass::ModelPass {
    uint32_t m_seq_len_dim;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareEncoderEmbeddingModel");

    explicit PrepareEncoderEmbeddingModel(uint32_t seq_len_dim) : m_seq_len_dim(seq_len_dim) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw::util
