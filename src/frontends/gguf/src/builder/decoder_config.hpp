// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "builder/decoder_metadata.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

class GraphEmitter;

// Internal configuration resolved from metadata, tensor shapes and architecture semantics.
// Extensions override ambiguous semantics through DecoderOptions before dependent plans are built.
struct DecoderConfig {
    DecoderConfig(const DecoderParameters& parameters, const std::unordered_map<std::string, ov::Tensor>& weights);

    // Sets the graph's RoPE, SWA and model-contract state and configures the weight transforms.
    void configure_graph(GraphEmitter& emitter, const GgufMetadata& metadata) const;

    std::string arch;
    ModelContract model_contract{};
    bool interleaved_q_gate = false;
    std::optional<bool> qkv_bias_override;
    std::optional<bool> output_bias_override;
    std::optional<bool> fused_qkv_override;

    // ---- core dimensions ----
    int n_layer = 0;
    int head_size = 0;
    int n_embd = 0;
    float rms_eps = 0.0f;
    bool embedding_model = false;
    int pooling_type = 0;
    bool causal_attention = true;
    bool layer_norm = false;
    bool relu_squared_ffn = false;

    // ---- auto-detected per-architecture structure ----
    bool has_qk_norm = false;
    bool qk_norm_after_rope = false;
    bool post_norm_only = false;
    bool qk_norm_full = false;  // norm width is n_head*head_size (OLMoE) rather than per-head
    bool has_qkv_bias = false;
    bool has_attn_out_bias = false;
    bool has_rope_freqs = false;
    bool has_moe_gate_bias = false;
    bool moe_softmax_weight = false;  // softmax AFTER top-k (gpt-oss) rather than before
    bool moe_swiglu_oai = false;      // OAI gated activation (gpt-oss)
    bool has_moe_expert_bias = false;
    bool has_sinks = false;
    bool has_swa = false;
    bool has_attn_post_norm = false;
    bool has_ffn_post_norm = false;
    std::string attn_post_norm_key{"post_attention_norm.weight"};
    std::string ffn_post_norm_key{"post_ffw_norm.weight"};
    bool is_geglu = false;
    bool has_v_norm = false;  // gemma4: V is also RMSNorm'd like K

    // muse-glimmer specifics
    bool has_attn_gate = false;        // sigmoid gate multiplied into the attention output
    bool scaleless_embd_norm = false;  // weightless RMSNorm on the token embeddings
    int rope_skip_period = 0;
    bool rope_on_swa_only = false;  // global (non-SWA) layers are NoPE
    float post_norm_eps = 0.0f;     // eps for post-attn/post-FFN norms (0 -> rms_eps)

    // ---- qwen35 (hybrid Gated-DeltaNet + full attention) ----
    int ssm_conv_kernel = 0;  // d_conv
    int ssm_state_size = 0;   // head_k_dim / head_v_dim
    int ssm_group_count = 0;  // num_k_heads
    int ssm_dt_rank = 0;      // num_v_heads
    int ssm_inner_size = 0;   // d_inner = num_v_heads * head_v_dim

    // Pre-norm key suffixes; GPT-OSS uses post_attention_norm for its pre-FFN norm.
    std::string attn_norm_key{"attn_norm.weight"};
    std::string ffn_norm_key{"ffn_norm.weight"};

    // ---- MoE / scalar metadata ----
    int n_expert = 0;
    int n_expert_used = 0;
    bool moe_sigmoid_gating = false;
    int expert_groups = 1;
    int expert_groups_used = 1;
    float embedding_scale = 1.0f;
    // Like llama.cpp's Gemma3/Gemma4, scale token lookups only; injected embeddings stay raw.
    bool embedding_scale_tokens_only = false;
    float residual_scale = 1.0f;
    float logit_scale = 1.0f;
    float attention_temperature_scale = 0.0f;
    float expert_weights_scale = 0.0f;  // 0 -> 1.0 no-op
    bool expert_weights_norm = false;   // renormalize top-K gate weights to sum to 1 (qwen3moe etc.)
    float attn_soft_cap = 0.0f;
    float final_logit_soft_cap = 0.0f;

    // ---- SWA / gemma4 ----
    float rope_freq_base_swa = 0.0f;
    int swa_layer_pattern = 2;
    int n_embd_per_layer = 0;  // gemma4: per-layer embedding projection dimension
    int shared_kv_layers = 0;  // gemma4: N trailing layers that share KV from earlier layers
    // Sliding-window length in tokens, when the GGUF metadata carries an explicit
    // "<arch>.attention.sliding_window" value (0 -> not present, e.g. an SWA layout described only
    // by sinks or a per-layer boolean pattern with no accompanying token count).
    int swa_window_size = 0;

    // ---- RoPE ----
    int rope_op_case = 0;
    RopeConfig rope_config{};
    bool use_per_op_rope = false;

    // ---- Per-layer hyperparameter accessors ----
    // Per-layer values are resolved once in the constructor (explicit layer overrides win over the
    // metadata-derived rules), so the topology loop stays declarative.

    // Whether attention layer `il` uses a sliding window.
    bool layer_is_swa(int il) const {
        return m_layers.at(il).swa;
    }
    bool layer_is_moe(int il) const {
        return m_layers.at(il).moe;
    }
    // True when layer `il` is a linear-attention (recurrent) layer.
    bool is_recurrent_layer(int il) const {
        return m_layers.at(il).recurrent;
    }
    int layer_head_size(int il) const {
        return m_layers.at(il).head_size;
    }
    int layer_n_head(int il) const {
        return m_layers.at(il).n_head;
    }
    // KV head count for layer `il` (some archs, e.g. Deci-style, vary head_count_kv per layer).
    int layer_n_head_kv(int il) const {
        return m_layers.at(il).n_head_kv;
    }
    // Attention (softmax) scale: an explicit scale wins, otherwise 1/sqrt(layer head size).
    float layer_kq_scale(int il) const {
        return m_layers.at(il).kq_scale;
    }
    // RoPE config for layer `il` (e.g. gemma4 SWA layers use a different freq_base / n_dims).
    const RopeConfig& layer_rope_config(int il) const {
        return m_layers.at(il).rope;
    }

private:
    struct LayerSettings {
        bool swa = false;
        bool moe = false;
        bool recurrent = false;
        int head_size = 0;
        int n_head = 0;
        int n_head_kv = 0;
        float kq_scale = 0.0f;
        RopeConfig rope{};
    };
    std::vector<LayerSettings> m_layers;
};

}  // namespace ov::frontend::gguf
