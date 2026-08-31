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

// Per-layer KV-cache routing, precomputed once for the whole stack.
//
// gemma4 shares KV across layers: the trailing `shared_kv_layers` layers have no K/V of their own
// and reuse the cache of the last own-KV layer of the SAME sliding-window type. Everything else
// leaves this at its defaults, where every layer owns its cache.
struct KvCachePlan {
    int n_own_kv = 0;        // first n_own_kv layers have their own KV cache
    int anchor_swa = -1;     // last own-KV sliding-window layer, or -1
    int anchor_global = -1;  // last own-KV global layer, or -1

    // Build the plan for `cfg`; mirrors llama.cpp's layer_reuse_cb.
    static KvCachePlan build(const DecoderConfig& cfg);
};

// The attention sublayer of one decoder layer: Q/K/V projections (+ optional biases and Q/K/V
// norms), RoPE, the KV-cache store, FLASH_ATTN_EXT, the optional sigmoid output gate, and the
// output projection (+ optional bias).
//
// `attn_norm` is the pre-attention normed hidden. Returns the sublayer output tensor name BEFORE
// the residual add, matching what gated_delta_net() returns for the recurrent layers, so the
// decoder's shared FFN tail consumes either interchangeably.
std::string attention(GraphEmitter& e,
                      const DecoderConfig& cfg,
                      const KvCachePlan& kv,
                      int il,
                      const std::string& attn_norm,
                      int64_t T);

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
