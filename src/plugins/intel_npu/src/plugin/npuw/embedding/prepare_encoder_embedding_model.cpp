// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_encoder_embedding_model.hpp"

#include "../logging.hpp"

bool ov::npuw::util::PrepareEncoderEmbeddingModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Preparing encoder (bidirectional) text-embedding model, seq_len_dim=" << m_seq_len_dim);
    LOG_BLOCK();

    // Non-autoregressive bidirectional encoder (e.g. BERT) embedding models are self-contained:
    //  - they build their own (bidirectional, non-causal) attention mask from `attention_mask`,
    //  - use learned absolute position embeddings (no RoPE), and
    //  - have no KV cache.
    // Therefore, unlike the autoregressive PrepareTextEmbeddingModel path, we must NOT inject
    // KV-cache parameters, RoPE position_ids, or a causal mask. The graph is left structurally
    // as-is (ReshapeToStatic later fixes the shapes to [1, L]); here we only sanity-check that
    // there is no autoregressive KV cache and re-validate shapes.
    for (const auto& param : model->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names()) {
            OPENVINO_ASSERT(name.find("past_key_values") == std::string::npos,
                            "PrepareEncoderEmbeddingModel: unexpected autoregressive KV-cache input '",
                            name,
                            "' in an encoder embedding model.");
        }
    }

    model->validate_nodes_and_infer_types();

    LOG_DEBUG("Done");
    return true;
}
