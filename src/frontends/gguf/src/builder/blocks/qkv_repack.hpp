// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "builder/decoder_config.hpp"
#include "builder/graph_emitter.hpp"

namespace ov::frontend::gguf::blocks {

// Split a fused blk.<il>.attn_qkv weight into separate q/k/v weight nodes registered under
// blk.<il>.attn_{q,k,v}.weight. Rows split as [n_head*head_size | n_head_kv*head_size x2].
void register_fused_qkv(GraphEmitter& e, const DecoderConfig& cfg, int il);

// Preserve the interleaved Q/gate weight for a joint projection.
void register_interleaved_q_gate(GraphEmitter& e, const DecoderConfig& cfg, int il);

}  // namespace ov::frontend::gguf::blocks
