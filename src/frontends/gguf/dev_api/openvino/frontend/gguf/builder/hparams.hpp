// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// The common decoder hyperparameters, read from the architecture-prefixed GGUF metadata keys.
// The port of llama.cpp's `llama_hparams`, with the same member spelling so an expression copied
// from a model file (`hparams.n_embd_head_v()`, `n_head_kv(il)`) reads the same here.
//
// GGUF names these keys after the architecture ("<arch>.block_count",
// "<arch>.attention.head_count", ...), so this is only meaningful for a file that names an LLM
// architecture. A non-decoder family (mmproj vision/audio) carries a different key layout entirely
// and should read what it needs through GgufMetadata directly rather than through this.
//
// Nothing here throws: a key the file omits leaves its field at the documented default, because
// GGUF metadata is untrusted and "absent" is a normal, meaningful state. A builder that cannot
// proceed without a value checks for it and says so itself.
struct GGUF_FRONTEND_API GgufHparams {
    // Read the "<arch>.*" keys for `arch` out of `meta`.
    GgufHparams(const GgufMetadata& meta, const std::string& arch);

    std::string arch;

    uint32_t n_embd = 0;
    uint32_t n_layer = 0;
    uint32_t n_ff = 0;
    uint32_t n_vocab = 0;
    uint32_t n_head_arr = 0;     // <arch>.attention.head_count
    uint32_t n_head_kv_arr = 0;  // <arch>.attention.head_count_kv
    uint32_t n_rot = 0;
    uint32_t n_embd_head_k_v = 0;  // <arch>.attention.key_length   (0 -> n_embd / n_head)
    uint32_t n_embd_head_v_v = 0;  // <arch>.attention.value_length (0 -> n_embd / n_head)
    uint32_t n_ctx_train = 0;

    float f_norm_rms_eps = 0.0f;
    float f_norm_eps = 0.0f;
    float rope_freq_base_train = 10000.0f;
    float rope_freq_scale_train = 1.0f;

    // MoE
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_shared = 0;
    uint32_t n_layer_dense_lead = 0;
    float expert_weights_scale = 0.0f;
    bool expert_weights_norm = false;

    // sliding-window attention
    uint32_t n_swa = 0;          // window size in tokens (0 -> none configured)
    uint32_t n_swa_pattern = 0;  // period; layer is SWA unless it is the last of each period
    std::vector<int32_t> swa_layers;

    // soft-caps (gemma2/3)
    float f_attn_logit_softcapping = 0.0f;
    float f_final_logit_softcapping = 0.0f;

    // scalar scales (minicpm)
    float f_embedding_scale = 1.0f;
    float f_residual_scale = 1.0f;
    float f_logit_scale = 1.0f;

    // per-layer KV head counts, when the file stores head_count_kv as an array
    std::vector<int64_t> n_head_kv_per_layer;

    // ---- llama.cpp-named accessors ----
    uint32_t n_head(int il = 0) const;
    uint32_t n_head_kv(int il = 0) const;
    uint32_t n_embd_head_k() const;
    uint32_t n_embd_head_v() const;
    // Total width of the K (resp. V) projection: n_head_kv(il) * n_embd_head_k().
    uint32_t n_embd_k_gqa(int il = 0) const;
    uint32_t n_embd_v_gqa(int il = 0) const;
    // Whether layer `il` uses a sliding window: an explicit per-layer array wins over the period,
    // which follows llama.cpp's set_swa_pattern(period, dense_first=false).
    bool is_swa(int il) const;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
