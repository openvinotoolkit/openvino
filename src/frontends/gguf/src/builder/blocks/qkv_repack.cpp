// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/qkv_repack.hpp"

#include <array>

#include "quant/weights.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

void register_fused_qkv(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const size_t n_q = static_cast<size_t>(cfg.n_head) * cfg.head_size;
    const size_t n_kv = static_cast<size_t>(cfg.n_head_kv) * cfg.head_size;
    auto parts = split_fused_qkv_extracted(p + "attn_qkv", e.weights(), e.qtypes(), n_q, n_kv, n_kv);
    const std::array<std::string, 3> names = {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"};
    const std::array<int64_t, 3> rows = {(int64_t)n_q, (int64_t)n_kv, (int64_t)n_kv};
    for (size_t i = 0; i < 3; ++i) {
        e.emit_weight_op(names[i], parts[i].tensors, parts[i].qtype, ps({1, 1, rows[i], cfg.n_embd}));
    }

    // Some fused-QKV archs (e.g. phi-3) also carry a single attn_qkv.bias; split it into
    // attn_{q,k,v}.bias the same way so attention()'s plain add_bias lookup finds them. Not every
    // fused-QKV arch has a bias, so only split when the fused tensor is actually present.
    if (e.weights().count(p + "attn_qkv.bias")) {
        auto bias_parts = split_fused_qkv_bias(p + "attn_qkv", e.weights(), n_q, n_kv, n_kv);
        const std::array<std::string, 3> bias_names = {p + "attn_q.bias", p + "attn_k.bias", p + "attn_v.bias"};
        for (size_t i = 0; i < 3; ++i) {
            e.weights()[bias_names[i]] = bias_parts[i];
        }
    }
}

void register_qwen35_q_gate(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto parts = split_interleaved_q_gate(p + "attn_q", e.weights(), e.qtypes(), static_cast<size_t>(cfg.head_size));
    const std::array<std::string, 2> names = {p + "attn_q.weight", p + "attn_gate.weight"};
    const int64_t rows = static_cast<int64_t>(cfg.n_head) * cfg.head_size;
    for (size_t i = 0; i < 2; ++i) {
        e.emit_weight_op(names[i], parts[i].tensors, parts[i].qtype, ps({1, 1, rows, cfg.n_embd}));
    }
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
