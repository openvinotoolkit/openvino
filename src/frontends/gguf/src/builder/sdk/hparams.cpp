// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Decoder hyperparameters, read from the architecture-prefixed GGUF metadata keys.
//
// Ground truth for the key spellings is llama.cpp's llama-arch.cpp (LLM_KV_* table); ground truth
// for the fallbacks is its llama_model::load_hparams.

#include "openvino/frontend/gguf/builder/hparams.hpp"

namespace ov {
namespace frontend {
namespace gguf {

GgufHparams::GgufHparams(const GgufMetadata& meta, const std::string& architecture) : arch(architecture) {
    const auto k = [&](const std::string& suffix) {
        return arch + "." + suffix;
    };

    meta.get_key(k("embedding_length"), n_embd);
    meta.get_key(k("block_count"), n_layer);
    meta.get_key(k("feed_forward_length"), n_ff);
    meta.get_key(k("context_length"), n_ctx_train);
    meta.get_key(k("attention.head_count"), n_head_arr);
    meta.get_key(k("attention.head_count_kv"), n_head_kv_arr);
    meta.get_key(k("rope.dimension_count"), n_rot);
    meta.get_key(k("attention.key_length"), n_embd_head_k_v);
    meta.get_key(k("attention.value_length"), n_embd_head_v_v);

    meta.get_key(k("attention.layer_norm_rms_epsilon"), f_norm_rms_eps);
    meta.get_key(k("attention.layer_norm_epsilon"), f_norm_eps);
    meta.get_key(k("rope.freq_base"), rope_freq_base_train);
    meta.get_key(k("rope.scaling.factor"), rope_freq_scale_train);

    meta.get_key(k("expert_count"), n_expert);
    meta.get_key(k("expert_used_count"), n_expert_used);
    meta.get_key(k("expert_shared_count"), n_expert_shared);
    meta.get_key(k("leading_dense_block_count"), n_layer_dense_lead);
    meta.get_key(k("expert_weights_scale"), expert_weights_scale);
    meta.get_key(k("expert_weights_norm"), expert_weights_norm);

    meta.get_key(k("attention.sliding_window"), n_swa);
    meta.get_key(k("attention.sliding_window_pattern"), n_swa_pattern);

    meta.get_key(k("attn_logit_softcapping"), f_attn_logit_softcapping);
    meta.get_key(k("final_logit_softcapping"), f_final_logit_softcapping);

    meta.get_key(k("embedding_scale"), f_embedding_scale);
    meta.get_key(k("residual_scale"), f_residual_scale);
    meta.get_key(k("logit_scale"), f_logit_scale);

    // head_count_kv is an ARRAY for architectures whose KV head count varies by layer; the scalar
    // read above leaves n_head_kv_arr at 0 in that case, so fall back to the array's first entry
    // and keep the whole array for the per-layer accessor.
    n_head_kv_per_layer = meta.get_int_array(k("attention.head_count_kv"));
    if (n_head_kv_arr == 0 && !n_head_kv_per_layer.empty()) {
        n_head_kv_arr = static_cast<uint32_t>(n_head_kv_per_layer[0]);
    }

    // Per-layer SWA flags, when the file spells the pattern out rather than as a period.
    for (auto v : meta.get_int_array(k("attention.sliding_window_layers"))) {
        swa_layers.push_back(static_cast<int32_t>(v));
    }

    // Vocabulary size is not its own key: it is the row count of the token list.
    n_vocab = static_cast<uint32_t>(meta.get_str_array("tokenizer.ggml.tokens").size());
}

uint32_t GgufHparams::n_head(int) const {
    return n_head_arr;
}

uint32_t GgufHparams::n_head_kv(int il) const {
    if (!n_head_kv_per_layer.empty() && il >= 0 && static_cast<size_t>(il) < n_head_kv_per_layer.size()) {
        return static_cast<uint32_t>(n_head_kv_per_layer[il]);
    }
    return n_head_kv_arr;
}

uint32_t GgufHparams::n_embd_head_k() const {
    if (n_embd_head_k_v != 0) {
        return n_embd_head_k_v;
    }
    return n_head_arr != 0 ? n_embd / n_head_arr : 0;
}

uint32_t GgufHparams::n_embd_head_v() const {
    if (n_embd_head_v_v != 0) {
        return n_embd_head_v_v;
    }
    return n_head_arr != 0 ? n_embd / n_head_arr : 0;
}

uint32_t GgufHparams::n_embd_k_gqa(int il) const {
    return n_head_kv(il) * n_embd_head_k();
}

uint32_t GgufHparams::n_embd_v_gqa(int il) const {
    return n_head_kv(il) * n_embd_head_v();
}

bool GgufHparams::is_swa(int il) const {
    // An explicit per-layer array wins over the period, matching llama.cpp's load order.
    if (!swa_layers.empty()) {
        return il >= 0 && static_cast<size_t>(il) < swa_layers.size() && swa_layers[il] != 0;
    }
    if (n_swa_pattern == 0) {
        return false;
    }
    // llama.cpp set_swa_pattern(period, dense_first=false): a layer is SWA unless it is the last
    // of each period.
    return (il % static_cast<int>(n_swa_pattern)) < (static_cast<int>(n_swa_pattern) - 1);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
