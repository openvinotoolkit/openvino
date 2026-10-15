// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "builder/decoder_config.hpp"
#include "builder/graph_emitter.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

// Feed-forward sublayers of the decoder block, mirroring llm_graph_context::build_ffn /
// build_moe_ffn. `prefix` is the layer prefix ("blk.<il>."), `ffn_norm` the pre-FFN normed hidden,
// and `T` the representative static token length. Each returns the sublayer output tensor name,
// before the residual add.

// Dense SwiGLU FFN (llama/qwen/phi3/minicpm). Supports the fused gate+up projection (phi-3, no
// ffn_gate.weight) and optional per-projection biases (pangu-embedded), both auto-detected from
// the weight table.
std::string dense_ffn(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const std::string& prefix,
                      const std::string& ffn_norm,
                      int64_t T);

// Dense GeGLU FFN (Gemma/Gemma2). Same layout as SwiGLU but uses GELU activation.
std::string geglu_ffn(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const std::string& prefix,
                      const std::string& ffn_norm,
                      int64_t T);

// Mixture-of-experts FFN (OLMoE / gpt-oss / qwen3moe), mirroring llm_graph_context::build_moe_ffn.
// Routing: logits = gate_inp·x; probs = softmax/identity; pick top-k experts; per-token expert
// matmuls via MUL_MAT_ID; gated activation; weighted sum over the used experts; plus the optional
// always-active shared experts.
std::string moe_ffn(GraphEmitter& e,
                    const DecoderConfig& cfg,
                    const std::string& prefix,
                    const std::string& ffn_norm,
                    int64_t T);

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
