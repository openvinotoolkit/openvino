// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "builder/arch_registry.hpp"
#include "builder/blocks/attention.hpp"
#include "builder/decoder_config.hpp"
#include "builder/graph_emitter.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

// Whole-model topology for configured dense, expert and hybrid decoder layers.
class DecoderBuilder : public ModelBuilder {
public:
    DecoderBuilder(const BuildContext& context, const DecoderParameters& parameters);

    std::shared_ptr<GgufGraph> build() override;

private:
    struct AttentionInputs {
        std::string positions = "inp_pos";
        bool last_query_rows = false;
    };

    // Model inputs shared by every layer (tokens, positions, masks, KV update index).
    void build_inputs();

    // Token embedding lookup plus the optional embedding scale / weightless norm.
    std::string build_embeddings();

    // gemma4: the per-layer input embeddings, pre-projected once before the layer loop.
    void build_per_layer_embeddings(const std::string& embd);

    // One decoder layer; `cur` is the layer input, returns the layer output.
    std::string build_layer(int il,
                            const std::string& cur,
                            const std::string& full_token_ref,
                            const AttentionInputs& attention_inputs);

    // gemma4: per-layer embedding injection appended after the FFN residual.
    std::string inject_per_layer_embedding(int il,
                                           const std::string& cur,
                                           const std::string& inpSA,
                                           const std::string& full_token_ref);

    // Final norm, lm_head, and the optional logit scale / soft-cap.
    std::string build_head(const std::string& cur);
    std::string build_gemma3n(const std::string& embeddings);
    std::string build_eagle3(const std::string& embeddings);
    std::string build_dflash(const std::string& embeddings);
    bool prunes_shared_kv_rows() const {
        return m_genai_adaptation && m_cfg.shared_kv_layers > 0 && !m_cfg.embedding_model;
    }
    int first_shared_kv_layer() const {
        return m_cfg.n_layer - m_cfg.shared_kv_layers;
    }
    bool selects_output_rows(int il) const {
        return il == m_cfg.n_layer - 1 && !m_cfg.embedding_model && !prunes_shared_kv_rows();
    }
    bool selects_per_layer_embedding_rows(int il) const {
        return selects_output_rows(il) || (prunes_shared_kv_rows() && il >= first_shared_kv_layer());
    }

    DecoderConfig m_cfg;
    GraphEmitter m_emit;
    blocks::KvCachePlan m_kv;
    bool m_genai_adaptation = false;
};

}  // namespace ov::frontend::gguf
