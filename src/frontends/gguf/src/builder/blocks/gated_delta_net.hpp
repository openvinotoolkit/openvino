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

// qwen35 linear-attention (Gated DeltaNet) layer.
// Mirrors llama.cpp src/models/qwen35.cpp build_layer_attn_linear + delta-net-base.cpp.
// `attn_norm` is the pre-attention normed hidden; returns the sublayer output BEFORE the residual
// add, i.e. exactly what the shared decoder tail expects in place of attn_out.
//
// The two recurrent states are plain model Parameters written through to Results, matching how the
// builder treats KV caches: the frontend always emits a STATELESS graph and leaves statefulness to
// the consumer. Unlike a KV cache these are OVERWRITTEN, not appended, so they carry no token axis
// and MakeStateful's Concat path does not apply to them.
std::string gated_delta_net(GraphEmitter& e, const DecoderConfig& cfg, int il, const std::string& attn_norm, int64_t T);

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
