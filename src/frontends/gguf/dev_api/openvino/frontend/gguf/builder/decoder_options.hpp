// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

enum class RopeMode { Normal, Neox, Interleaved };

enum class NormalizationKind { Rms, Layer };
enum class FfnActivation { SwiGlu, GeGlu, ReluSquared };
enum class RecurrentKind { None, GatedDeltaNet };
enum class ExpertRouting { Softmax = 1, Sigmoid = 2 };

struct GGUF_FRONTEND_API DecoderShape {
    int layers = 0;
    int embedding = 0;
    int query_heads = 0;
    int kv_heads = 0;
    int head_size = 0;
    std::vector<uint32_t> kv_heads_per_layer;
    int per_layer_embedding = 0;
    int shared_kv_layers = 0;
};

struct GGUF_FRONTEND_API DecoderNormalization {
    NormalizationKind kind = NormalizationKind::Rms;
    float epsilon = 0.0f;
    float post_epsilon = 0.0f;
    bool post_only = false;
    bool qk_after_rope = false;
    std::optional<bool> qk_full_width;
    std::optional<bool> qk_norm;
    bool value_norm = false;
    bool embedding_norm = false;
    std::string attention_key = "attn_norm.weight";
    std::string ffn_key = "ffn_norm.weight";
    std::string attention_post_key = "post_attention_norm.weight";
    std::string ffn_post_key = "post_ffw_norm.weight";
    std::optional<bool> attention_post;
    std::optional<bool> ffn_post;
};

struct GGUF_FRONTEND_API DecoderScales {
    float embedding = 1.0f;
    float residual = 1.0f;
    float logits = 1.0f;
    bool token_embeddings_only = false;
    float attention = 0.0f;
    float attention_temperature = 0.0f;
    float attention_soft_cap = 0.0f;
    float logits_soft_cap = 0.0f;
};

struct GGUF_FRONTEND_API DecoderAttention {
    RopeMode rope_mode = RopeMode::Normal;
    RopeConfig rope{};
    RopeConfig sliding_rope{};
    float yarn_log_multiplier = 0.0f;
    bool sliding_window = false;
    int window_size = 0;
    int sliding_period = 2;
    std::vector<int32_t> sliding_layers;
    int sliding_head_size = 0;
    bool rope_on_sliding_only = false;
    int rope_skip_period = 0;
    bool sliding_on_attention_layers = false;
    bool interleaved_q_gate = false;
    std::optional<bool> fused_qkv;
    std::optional<bool> output_gate;
    std::optional<bool> qkv_bias;
    std::optional<bool> output_bias;
    std::optional<bool> sinks;
};

struct GGUF_FRONTEND_API DecoderExperts {
    int count = 0;
    int used = 0;
    int dense_lead = 0;
    int layer_step = 1;
    int groups = 1;
    int groups_used = 1;
    ExpertRouting routing = ExpertRouting::Softmax;
    float weights_scale = 0.0f;
    bool normalize_weights = false;
    std::optional<bool> enabled;
    std::optional<bool> softmax_after_topk;
    std::optional<bool> swiglu_oai;
};

struct GGUF_FRONTEND_API DecoderHybrid {
    RecurrentKind kind = RecurrentKind::None;
    int conv_kernel = 0;
    int state_size = 0;
    int group_count = 0;
    int time_step_rank = 0;
    int inner_size = 0;
    int full_attention_interval = 4;
    int nextn_layers = 0;
    std::vector<int32_t> recurrent_layers;
};

// Sparse overrides describe layer inputs; execution and cache plans remain internal.
struct GGUF_FRONTEND_API DecoderLayerOverrides {
    std::optional<int> query_heads;
    std::optional<int> kv_heads;
    std::optional<int> head_size;
    std::optional<float> attention_scale;
    std::optional<RopeConfig> rope;
    std::optional<bool> sliding_window;
    std::optional<bool> recurrent;
    std::optional<bool> experts;
};

struct GGUF_FRONTEND_API DecoderParameters {
    std::string architecture;
    DecoderShape dimensions;
    DecoderNormalization normalization;
    DecoderScales scales;
    DecoderAttention attention;
    DecoderExperts experts;
    DecoderHybrid hybrid;
    FfnActivation activation = FfnActivation::SwiGlu;
    bool embedding_model = false;
    int pooling_type = 0;
    bool causal_attention = true;
    ModelContract model_contract{};
    std::vector<DecoderLayerOverrides> layers;
};

// Read generic GGUF fields using explicitly selected semantics and defaults.
GGUF_FRONTEND_API DecoderParameters read_decoder_parameters(const GgufMetadata& metadata,
                                                            DecoderParameters defaults = {});

// Compatibility overrides; family defaults are selected by the builder before these apply.
struct GGUF_FRONTEND_API DecoderOptions {
    std::optional<NormalizationKind> normalization;
    std::optional<float> norm_epsilon;
    std::optional<float> embedding_scale;
    std::optional<float> residual_scale;
    std::optional<float> logit_scale;
    std::optional<float> attention_scale;
    std::optional<bool> qk_norm_after_rope;
    std::optional<bool> post_norm_only;
    std::optional<bool> normalize_expert_weights;
    std::optional<bool> geglu;
    std::optional<bool> value_norm;
    std::optional<bool> embedding_norm;
    std::optional<bool> rope_on_swa_only;
    std::optional<float> post_norm_epsilon;
    std::optional<float> swa_rope_frequency_base;
    std::optional<int> swa_rope_dimensions;
    std::optional<int> sliding_window;
    std::optional<int> rope_skip_period;
};

// Value snapshot; changes do not affect the shared decoder configuration.
struct GGUF_FRONTEND_API DecoderDimensions {
    int layers;
    int embedding;
    float norm_epsilon;
};

// Per-layer snapshot from decoder_attention configuration, including KV heads and SWA RoPE.
struct GGUF_FRONTEND_API DecoderLayerParameters {
    int query_heads;
    int kv_heads;
    int head_size;
    float attention_scale;
    RopeConfig rope;
    bool sliding_window;
    bool recurrent;
};

}  // namespace ov::frontend::gguf
