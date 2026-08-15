// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Everything the decoder topology needs to know about ONE model, resolved once up front.
//
// This is the GGUF frontend's equivalent of llama.cpp's llama_hparams plus its per-arch
// load_arch_hparams: a flat, already-decided description of the model, so the topology builder in
// arch/decoder_builder.cpp never re-derives anything and reads declaratively.
//
// Almost every field is DETECTED, not hard-coded per architecture: the presence of a layer-0
// weight tensor decides whether the model has QK-norm, projection biases, fused QKV, MoE routing,
// attention sinks, post-norms and so on. That is what lets a new same-family architecture be
// enabled by adding its name to arch_registry.cpp and writing no code at all. Only a handful of
// genuinely tensor-table-ambiguous properties (GeGLU vs SwiGLU, gemma4's V-norm) fall back to an
// architecture-name check; prefer weight presence when adding a new one.
struct DecoderConfig {
    // Resolve the whole description from the parsed metadata (already normalized by
    // config_from_meta) and the parser's tensor table.
    DecoderConfig(const std::map<std::string, GGUFMetaData>& config,
                  const std::unordered_map<std::string, ov::Tensor>& weights);

    std::string arch;

    // ---- core dimensions ----
    int n_layer = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_size = 0;
    int n_embd = 0;
    float rms_eps = 0.0f;
    std::vector<uint32_t> n_head_kv_per_layer;  // non-empty when head_count_kv varies by layer

    // ---- auto-detected per-architecture structure ----
    bool has_qk_norm = false;
    bool qk_norm_full = false;  // norm width is n_head*head_size (OLMoE) rather than per-head
    bool has_qkv_bias = false;
    bool has_attn_out_bias = false;
    bool has_rope_freqs = false;
    bool has_fused_qkv = false;
    bool is_moe = false;
    bool has_moe_gate_bias = false;
    bool moe_softmax_weight = false;  // softmax AFTER top-k (gpt-oss) rather than before
    bool moe_swiglu_oai = false;      // OAI gated activation (gpt-oss)
    bool has_moe_expert_bias = false;
    bool has_sinks = false;
    bool has_swa = false;
    bool has_attn_post_norm = false;
    bool has_ffn_post_norm = false;
    bool is_geglu = false;
    bool has_v_norm = false;  // gemma4: V is also RMSNorm'd like K

    // muse-glimmer specifics
    bool has_attn_gate = false;        // sigmoid gate multiplied into the attention output
    bool scaleless_embd_norm = false;  // weightless RMSNorm on the token embeddings
    bool rope_on_swa_only = false;     // global (non-SWA) layers are NoPE
    float post_norm_eps = 0.0f;        // eps for post-attn/post-FFN norms (0 -> rms_eps)

    // ---- qwen35 (hybrid Gated-DeltaNet + full attention) ----
    bool is_qwen35 = false;
    int ssm_conv_kernel = 0;                     // d_conv
    int ssm_state_size = 0;                      // head_k_dim / head_v_dim
    int ssm_group_count = 0;                     // num_k_heads
    int ssm_dt_rank = 0;                         // num_v_heads
    int ssm_inner_size = 0;                      // d_inner = num_v_heads * head_v_dim
    int full_attn_interval = 0;                  // full attention when (il + 1) % interval == 0
    int n_layer_nextn = 0;                       // trailing MTP blocks, not executed
    std::vector<int32_t> rope_sections;          // M-RoPE per-axis section widths
    std::vector<int32_t> recurrent_layer_flags;  // explicit per-layer recurrent flags

    // Norm key suffixes; overridden for archs that use non-standard naming (exaone4, gpt-oss).
    std::string attn_norm_key{"attn_norm.weight"};
    std::string ffn_norm_key{"ffn_norm.weight"};

    // ---- MoE / scalar metadata ----
    int n_expert = 0;
    int n_expert_used = 0;
    int n_dense_lead = 0;
    int n_expert_shared = 0;
    float embedding_scale = 1.0f;
    float residual_scale = 1.0f;
    float logit_scale = 1.0f;
    float attention_scale = 0.0f;       // 0 -> 1/sqrt(head_size)
    float expert_weights_scale = 0.0f;  // 0 -> 1.0 no-op
    float attn_soft_cap = 0.0f;
    float final_logit_soft_cap = 0.0f;

    // ---- SWA / gemma4 ----
    float rope_freq_base_swa = 0.0f;
    int swa_layer_pattern = 2;
    std::vector<int32_t> swa_layer_flags;  // gemma4: per-layer SWA flags (1=SWA, 0=global)
    int n_embd_per_layer = 0;              // gemma4: per-layer embedding projection dimension
    int shared_kv_layers = 0;              // gemma4: N trailing layers that share KV from earlier layers
    int head_size_swa = 0;                 // gemma4: head size for SWA layers (differs from global)
    int rope_dim_swa = 0;                  // gemma4: rope dims for SWA layers

    // ---- RoPE ----
    int rope_op_case = 0;
    RopeConfig rope_config{};
    RopeConfig rope_config_swa{};
    bool use_per_op_rope = false;

    // ---- Per-layer hyperparameter accessors ----
    // Centralize the per-layer derivations that vary by architecture (SWA layers, variable head
    // counts). They are the single source of truth, so the topology loop stays declarative.
    // If a new arch adds a per-layer dimension, extend these rather than inlining a ternary.

    // Whether attention layer `il` uses a sliding window. gemma4 carries an explicit per-layer
    // boolean array; gpt-oss/gemma3/smollm3 use a period (il is SWA unless it is the last in each
    // period, matching llama.cpp set_swa_pattern(period, dense_first=false)).
    bool layer_is_swa(int il) const;

    // Head size for layer `il`. gemma4 SWA layers use a smaller head size than global layers.
    int layer_head_size(int il) const;

    // KV head count for layer `il` (some archs, e.g. Deci-style, vary head_count_kv per layer).
    int layer_n_head_kv(int il) const;

    // Attention (softmax) scale for layer `il`. An explicit metadata scale wins; otherwise use
    // 1/sqrt(layer_head_size(il)) so SWA and global layers each get the head-size-correct scale.
    float layer_kq_scale(int il) const;

    // RoPE config for layer `il` (gemma4 SWA layers rope with a different freq_base / n_dims).
    RopeConfig layer_rope_config(int il) const;

    // True when layer `il` is a linear-attention (recurrent) layer. An explicit per-layer flag
    // array wins over the interval rule, matching llama.cpp's load order.
    bool is_recurrent_layer(int il) const;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
