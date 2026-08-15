// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/qkv_repack.hpp"

#include <array>
#include <unordered_map>

#include "quant/weights.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

namespace {

// Re-key the sliced tensors from "<src_base>.<part>.weight" to "<dst_base>.weight" so
// emit_weight_op's sub-key extraction (weight/scales/zp) and translate_weight's make_weight_node
// see the node's own base name.
std::unordered_map<std::string, ov::Tensor> rekey(const std::unordered_map<std::string, ov::Tensor>& extracted,
                                                  const std::string& dst_base) {
    std::unordered_map<std::string, ov::Tensor> out;
    for (const auto& kv : extracted) {
        auto dot = kv.first.rfind('.');
        out[dst_base + "." + kv.first.substr(dot + 1)] = kv.second;
    }
    return out;
}

}  // namespace

void register_fused_qkv(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const size_t n_q = static_cast<size_t>(cfg.n_head) * cfg.head_size;
    const size_t n_kv = static_cast<size_t>(cfg.n_head_kv) * cfg.head_size;
    auto parts = split_fused_qkv_extracted(p + "attn_qkv", e.weights(), e.qtypes(), n_q, n_kv, n_kv);
    const std::array<std::string, 3> names = {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"};
    const std::array<int64_t, 3> rows = {(int64_t)n_q, (int64_t)n_kv, (int64_t)n_kv};
    for (size_t i = 0; i < 3; ++i) {
        const std::string base = p + "attn_" + std::string(1, "qkv"[i]);  // blk.N.attn_q etc.
        e.emit_weight_op(names[i], rekey(parts[i].extracted, base), parts[i].qtype, ps({1, 1, rows[i], cfg.n_embd}));
    }
}

void register_qwen35_q_gate(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto parts = split_interleaved_q_gate(p + "attn_q", e.weights(), e.qtypes(), static_cast<size_t>(cfg.head_size));
    const std::array<std::string, 2> names = {p + "attn_q.weight", p + "attn_gate.weight"};
    const std::array<std::string, 2> suffix = {"q", "gate"};
    const int64_t rows = static_cast<int64_t>(cfg.n_head) * cfg.head_size;
    for (size_t i = 0; i < 2; ++i) {
        const std::string base = p + "attn_" + suffix[i];
        e.emit_weight_op(names[i], rekey(parts[i].extracted, base), parts[i].qtype, ps({1, 1, rows, cfg.n_embd}));
    }
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
