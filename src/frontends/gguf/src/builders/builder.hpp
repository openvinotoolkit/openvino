// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_reader.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief Build the qwen3 transformer graph from a parsed GGUF model.
///
/// Weight tensors are emitted as raw gguf_* Constants and consumed by FullyConnectedCompressed
/// nodes (no in-graph dequantization). Throws if a required tensor or metadata key is missing.
std::shared_ptr<ov::Model> build_qwen3_model(const GGUFReader& reader);

/// \brief Build the qwen35 (Qwen3.5 dense) transformer graph from a parsed GGUF model.
///
/// qwen35 is a hybrid decoder: its token mixer alternates between a linear Gated-DeltaNet SSM and
/// a full gated attention (every `full_attention_interval`-th layer), same as qwen35moe, but the
/// FFN block is a standard dense SwiGLU MLP (ffn_gate / ffn_up / ffn_down) rather than a
/// mixture-of-experts block. Throws if a required tensor or metadata key is missing.
std::shared_ptr<ov::Model> build_qwen35_model(const GGUFReader& reader);

/// \brief Build the qwen35moe (Qwen3.5/3.6-MoE) transformer graph from a parsed GGUF model.
///
/// qwen35moe is a hybrid model: every layer has a MoE FFN block (256 routed experts + 1 shared
/// expert), and its token mixer alternates between a linear Gated-DeltaNet SSM and a full gated
/// attention (every `full_attention_interval`-th layer). Expert weights are emitted as raw gguf_*
/// block Constants consumed by MOECompressed (no in-graph dequantization). Throws if a required
/// tensor or metadata key is missing.
std::shared_ptr<ov::Model> build_qwen35moe_model(const GGUFReader& reader);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
