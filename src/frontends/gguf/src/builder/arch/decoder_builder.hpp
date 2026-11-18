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
    // Model inputs shared by every layer (tokens, positions, masks, KV update index).
    void build_inputs();

    // Token embedding lookup plus the optional embedding scale / weightless norm.
    std::string build_embeddings();

    // gemma4: the per-layer input embeddings, pre-projected once before the layer loop.
    void build_per_layer_embeddings(const std::string& embd);

    // One decoder layer; `cur` is the layer input, returns the layer output.
    std::string build_layer(int il, const std::string& cur);

    // gemma4: per-layer embedding injection appended after the FFN residual.
    std::string inject_per_layer_embedding(int il, const std::string& cur, const std::string& inpSA);

    // Final norm, lm_head, and the optional logit scale / soft-cap.
    std::string build_head(const std::string& cur);
    bool selects_output_rows(int il) const {
        return il == m_cfg.n_layer - 1 && !m_cfg.embedding_model;
    }

    DecoderConfig m_cfg;
    GraphEmitter m_emit;
    blocks::KvCachePlan m_kv;
};

}  // namespace ov::frontend::gguf
