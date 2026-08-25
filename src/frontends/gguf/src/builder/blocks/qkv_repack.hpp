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

// Weight-level repacking that happens before any op is emitted for a layer, so the rest of the
// attention path only ever sees the plain blk.<il>.attn_{q,k,v}.weight / attn_gate.weight nodes and
// stays architecture-agnostic.

// Split a fused blk.<il>.attn_qkv weight into separate q/k/v weight nodes registered under
// blk.<il>.attn_{q,k,v}.weight. Rows split as [n_head*head_size | n_head_kv*head_size x2].
void register_fused_qkv(GraphEmitter& e, const DecoderConfig& cfg, int il);

// qwen35 full-attention layers: attn_q packs query and attention-output gate interleaved per head
// ([q_h0 | gate_h0 | q_h1 | ...], stride 2*head_size). De-interleave into the plain
// blk.N.attn_q.weight / blk.N.attn_gate.weight the shared attention path expects, so the graph
// never sees the interleaving. llama.cpp does the same split with two strided views
// (src/models/qwen35.cpp build_layer_attn).
void register_qwen35_q_gate(GraphEmitter& e, const DecoderConfig& cfg, int il);

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
