// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/qkv_repack.hpp"

#include <array>

#include "openvino/core/except.hpp"

namespace ov::frontend::gguf::blocks {

void register_fused_qkv(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const size_t n_q = static_cast<size_t>(cfg.layer_n_head(il)) * cfg.layer_head_size(il);
    const size_t n_kv = static_cast<size_t>(cfg.layer_n_head_kv(il)) * cfg.layer_head_size(il);
    const std::string source = p + "attn_qkv.weight";
    OPENVINO_ASSERT(e.weight_rows(source) == static_cast<int64_t>(n_q + 2 * n_kv),
                    "[GGUF] fused qkv row mismatch for ",
                    source);
    const std::array<size_t, 3> starts = {0, n_q, n_q + n_kv};
    const std::array<size_t, 3> lengths = {n_q, n_kv, n_kv};
    const auto split = [&](const std::string& fused, const std::string& suffix) {
        const std::array<std::string, 3> names = {p + "attn_q" + suffix, p + "attn_k" + suffix, p + "attn_v" + suffix};
        for (size_t i = 0; i < names.size(); ++i) {
            e.derive_weight(
                names[i],
                "GGML_OP_VIEW",
                {fused},
                3,
                {{"view_slice",
                  std::vector<int64_t>{0, static_cast<int64_t>(starts[i]), static_cast<int64_t>(lengths[i])}}});
        }
    };
    split(source, ".weight");

    const std::string bias_source = p + "attn_qkv.bias";
    if (e.has_weight(bias_source)) {
        const auto& shape = e.weight_shape(bias_source);
        OPENVINO_ASSERT(shape.size() == 1 && shape[0] == n_q + 2 * n_kv,
                        "[GGUF] fused qkv bias row mismatch for ",
                        bias_source);
        split(bias_source, ".bias");
    }
}

void register_interleaved_q_gate(GraphEmitter& e, const DecoderConfig& cfg, int il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    e.add_weight_from(p + "attn_q_gate.weight", p + "attn_q");
}

}  // namespace ov::frontend::gguf::blocks
