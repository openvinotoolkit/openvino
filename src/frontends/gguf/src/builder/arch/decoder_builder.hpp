// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "builder/blocks/attention.hpp"
#include "builder/decoder_config.hpp"
#include "builder/graph_emitter.hpp"
#include "builder/model_builder.hpp"
#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Whole-model builder for the causal decoder family: the "llama family" of dense and MoE
// decoder-only transformers (llama-3, qwen2/2.5/3, phi-3, minicpm, gemma 1-4, gpt-oss, OLMoE,
// qwen3moe, the qwen35 hybrid Gated-DeltaNet stack, ...).
//
// ONE builder covers all of them. Per-architecture differences are resolved into DecoderConfig by
// structural detection on the GGUF tensor table, so enabling a same-family architecture is a name
// in arch_registry.cpp and no code here. This is a deliberate departure from llama.cpp's
// one-file-per-architecture layout, which exists there because each architecture must enumerate
// its tensors by hand; the GGUF frontend derives them instead.
//
// Emits the STATELESS graph: KV caches are plain model Parameters written by GGML_OP_SET_ROWS and
// read back as Results, and every activation keeps ggml's rank-4 shape. A caller that wants an
// OpenVINO KV cache registers ov::frontend::gguf::pass::MakeStateful as a transformation extension.
class DecoderBuilder : public ModelBuilder {
public:
    DecoderBuilder(const std::map<std::string, GGUFMetaData>& config,
                   std::unordered_map<std::string, ov::Tensor>& weights,
                   std::unordered_map<std::string, gguf_tensor_type>& qtypes);

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

    DecoderConfig m_cfg;
    GraphEmitter m_emit;
    blocks::KvCachePlan m_kv;

    // Per-node output shapes are STATIC, like the cgraph decoder (which builds the graph for a
    // concrete token length). We use a representative token length T; the translators emit dynamic
    // reshapes (-1 / 0) where needed, and MakeStateful + the dynamic input Parameters carry the
    // real dynamic-ness. T affects only the per-node shape metadata.
    static constexpr int64_t T = 1;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
