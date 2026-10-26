// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/decoder_config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/arch_registry.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {
namespace {
DecoderParameters legacy_parameters(const std::map<std::string, GGUFMetaData>& config,
                                    std::optional<RopeMode> rope,
                                    const DecoderOptions& options) {
    auto parameters = decoder_parameters_from_config(config);
    if (rope)
        parameters.attention.rope_mode = *rope;
    apply_decoder_options(parameters, options);
    return parameters;
}

void positive(float value, const char* name) {
    OPENVINO_ASSERT(std::isfinite(value) && value > 0, "[GGUF] ", name, " must be finite and positive");
}

void validate_rope(const RopeConfig& rope, int head_size) {
    OPENVINO_ASSERT(rope.n_dims >= 0 && rope.n_dims % 2 == 0 && rope.n_dims <= head_size,
                    "[GGUF] RoPE dimensions must be nonnegative, even and fit the attention head");
    if (rope.n_dims > 0) {
        positive(rope.freq_base, "RoPE frequency base");
        positive(rope.freq_scale, "RoPE frequency scale");
    }
}
}  // namespace

DecoderConfig::DecoderConfig(const std::map<std::string, GGUFMetaData>& config,
                             const std::unordered_map<std::string, ov::Tensor>& weights,
                             std::optional<RopeMode> rope,
                             const DecoderOptions& options)
    : DecoderConfig(legacy_parameters(config, rope, options), weights) {}

DecoderConfig::DecoderConfig(const DecoderParameters& p, const std::unordered_map<std::string, ov::Tensor>& weights) {
    const auto has = [&weights](const std::string& name) {
        return weights.count(name) > 0;
    };
    arch = p.architecture;
    model_contract = p.model_contract;
    n_layer = p.dimensions.layers;
    n_embd = p.dimensions.embedding;
    n_head = p.dimensions.query_heads;
    n_head_kv = p.dimensions.kv_heads;
    head_size = p.dimensions.head_size;
    n_head_kv_per_layer = p.dimensions.kv_heads_per_layer;
    n_embd_per_layer = p.dimensions.per_layer_embedding;
    shared_kv_layers = p.dimensions.shared_kv_layers;
    n_layer_nextn = p.hybrid.nextn_layers;
    OPENVINO_ASSERT(n_layer > 0 && n_embd > 0 && n_head > 0 && n_head_kv >= 0 && head_size > 0,
                    "[GGUF] invalid decoder dimensions");
    OPENVINO_ASSERT(n_layer_nextn >= 0 && n_layer_nextn < n_layer,
                    "[GGUF] invalid number of NextN blocks: ",
                    n_layer_nextn);
    const int stored_layers = n_layer;
    n_layer -= n_layer_nextn;
    const auto valid_layer_count = [&](size_t count) {
        return count == 0 || count == static_cast<size_t>(n_layer) || count == static_cast<size_t>(stored_layers);
    };
    OPENVINO_ASSERT(valid_layer_count(p.layers.size()) && valid_layer_count(n_head_kv_per_layer.size()) &&
                        valid_layer_count(p.attention.sliding_layers.size()) &&
                        valid_layer_count(p.hybrid.recurrent_layers.size()),
                    "[GGUF] per-layer settings must describe every decoder layer");
    layers = p.layers;
    rms_eps = p.normalization.epsilon;
    positive(rms_eps, "normalization epsilon");
    layer_norm = p.normalization.kind == NormalizationKind::Layer;
    relu_squared_ffn = p.activation == FfnActivation::ReluSquared;
    is_geglu = p.activation == FfnActivation::GeGlu;
    embedding_model = p.embedding_model;
    pooling_type = p.pooling_type;
    causal_attention = p.causal_attention;
    OPENVINO_ASSERT(!embedding_model || (pooling_type >= 0 && pooling_type <= 3),
                    "[GGUF] unsupported embedding pooling type: ",
                    pooling_type);
    qk_norm_after_rope = p.normalization.qk_after_rope;
    post_norm_only = p.normalization.post_only;
    has_v_norm = p.normalization.value_norm;
    scaleless_embd_norm = p.normalization.embedding_norm;
    post_norm_eps = p.normalization.post_epsilon;
    OPENVINO_ASSERT(post_norm_eps == 0.0f || (std::isfinite(post_norm_eps) && post_norm_eps > 0),
                    "[GGUF] post-norm epsilon must be finite and positive");
    attn_norm_key = p.normalization.attention_key;
    ffn_norm_key = p.normalization.ffn_key;
    attn_post_norm_key = p.normalization.attention_post_key;
    ffn_post_norm_key = p.normalization.ffn_post_key;
    is_qwen35 = p.hybrid.kind == RecurrentKind::GatedDeltaNet;
    ssm_conv_kernel = p.hybrid.conv_kernel;
    ssm_state_size = p.hybrid.state_size;
    ssm_group_count = p.hybrid.group_count;
    ssm_dt_rank = p.hybrid.time_step_rank;
    ssm_inner_size = p.hybrid.inner_size;
    full_attn_interval = p.hybrid.full_attention_interval;
    recurrent_layer_flags = p.hybrid.recurrent_layers;
    n_expert = p.experts.count;
    n_expert_used = p.experts.used;
    n_dense_lead = p.experts.dense_lead;
    moe_layer_step = std::max(1, p.experts.layer_step);
    expert_groups = std::max(1, p.experts.groups);
    expert_groups_used = std::max(1, p.experts.groups_used);
    moe_sigmoid_gating = p.experts.routing == ExpertRouting::Sigmoid;
    expert_weights_scale = p.experts.weights_scale;
    expert_weights_norm = p.experts.normalize_weights;
    embedding_scale = p.scales.embedding;
    residual_scale = p.scales.residual;
    logit_scale = p.scales.logits;
    embedding_scale_tokens_only = p.scales.token_embeddings_only;
    attention_scale = p.scales.attention;
    attention_temperature_scale = p.scales.attention_temperature;
    attn_soft_cap = p.scales.attention_soft_cap;
    final_logit_soft_cap = p.scales.logits_soft_cap;
    for (const auto value : {embedding_scale,
                             residual_scale,
                             logit_scale,
                             attention_scale,
                             attention_temperature_scale,
                             attn_soft_cap,
                             final_logit_soft_cap,
                             expert_weights_scale})
        OPENVINO_ASSERT(std::isfinite(value), "[GGUF] decoder scales must be finite");
    head_size_swa = p.attention.sliding_head_size;
    rope_dim_swa = p.attention.sliding_rope.n_dims;
    rope_freq_base_swa = p.attention.sliding_rope.freq_base;
    swa_window_size = p.attention.window_size;
    swa_layer_pattern = p.attention.sliding_period;
    swa_layer_flags = p.attention.sliding_layers;
    sliding_on_attention_layers = p.attention.sliding_on_attention_layers;
    has_swa = p.attention.sliding_window || swa_window_size > 0 || !swa_layer_flags.empty();
    rope_on_swa_only = p.attention.rope_on_sliding_only;
    rope_skip_period = p.attention.rope_skip_period;
    OPENVINO_ASSERT(rope_skip_period >= 0, "[GGUF] RoPE skip period must be nonnegative");
    OPENVINO_ASSERT(swa_window_size >= 0 && swa_layer_pattern >= 0, "[GGUF] invalid sliding-window settings");
    rope_config = p.attention.rope;
    rope_config_swa = p.attention.sliding_rope;
    if (rope_config_swa.n_dims == 0)
        rope_config_swa = rope_config;
    rope_dim_swa = rope_config_swa.n_dims;
    rope_freq_base_swa = rope_config_swa.freq_base;
    OPENVINO_ASSERT(attention_temperature_scale == 0.0f || rope_config.n_ctx_orig > 0,
                    "[GGUF] Attention temperature scaling requires a positive original context length");
    const float yarn_log_mul = p.attention.yarn_log_multiplier;
    if (rope_config.ext_factor != 0.0f && yarn_log_mul != 0.0f && rope_config.freq_scale < 1.0f) {
        const float magnitude = 1.0f + 0.1f * yarn_log_mul * std::log(1.0f / rope_config.freq_scale);
        rope_config.attn_factor /= magnitude;
        rope_config_swa.attn_factor /= magnitude;
    }
    rope_op_case = p.attention.rope_mode == RopeMode::Interleaved ? ROPE_OP_CASE_IMROPE
                   : p.attention.rope_mode == RopeMode::Neox      ? ROPE_OP_CASE_NEOX
                                                                  : ROPE_OP_CASE_NORMAL;
    if (rope_op_case == ROPE_OP_CASE_IMROPE) {
        rope_config.is_imrope = true;
        rope_sections.assign(rope_config.sections.begin(), rope_config.sections.end());
        rope_config_swa.is_imrope = true;
        rope_config_swa.sections = rope_config.sections;
    }
    OPENVINO_ASSERT(rope_config.is_imrope == (rope_op_case == ROPE_OP_CASE_IMROPE),
                    "[GGUF] RoPE configuration must match the selected position layout");
    std::string attention_probe = "blk.0.";
    int attention_layer = 0;
    for (int layer = 0; layer < n_layer; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        if (has(prefix + "attn_q_norm.weight")) {
            attention_probe = prefix;
            attention_layer = layer;
            break;
        }
    }
    has_qk_norm = p.normalization.qk_norm.value_or(has(attention_probe + "attn_q_norm.weight"));
    if (has_qk_norm && has(attention_probe + "attn_q_norm.weight")) {
        const auto& shape = weights.at(attention_probe + "attn_q_norm.weight").get_shape();
        OPENVINO_ASSERT(!shape.empty(), "[GGUF] empty Q normalization weight");
        qk_norm_full = shape[0] != static_cast<size_t>(layer_head_size(attention_layer));
    }
    qk_norm_full = p.normalization.qk_full_width.value_or(qk_norm_full);
    const int probe = (std::max(0, n_dense_lead) / moe_layer_step + 1) * moe_layer_step - 1;
    const auto expert_prefix = "blk." + std::to_string(probe) + ".";
    is_moe = p.experts.enabled.value_or(has(expert_prefix + "ffn_gate_exps.weight") ||
                                        has(expert_prefix + "ffn_gate_up_exps.weight"));
    has_moe_gate_bias = has(expert_prefix + "ffn_gate_inp.bias");
    has_moe_expert_bias = has(expert_prefix + "ffn_gate_exps.bias");
    moe_softmax_weight = p.experts.softmax_after_topk.value_or(has_moe_gate_bias);
    moe_swiglu_oai = p.experts.swiglu_oai.value_or(has_moe_gate_bias);
    has_sinks = p.attention.sinks.value_or(has("blk.0.attn_sinks.weight"));
    has_swa = has_swa || has_sinks;
    periodic_swa = has_swa;
    interleaved_q_gate = p.attention.interleaved_q_gate;
    qkv_bias_override = p.attention.qkv_bias;
    output_bias_override = p.attention.output_bias;
    fused_qkv_override = p.attention.fused_qkv;
    output_gate_override = p.attention.output_gate;
    has_fused_qkv = p.attention.fused_qkv.value_or(has("blk.0.attn_qkv.weight"));
    has_qkv_bias = p.attention.qkv_bias.value_or(has("blk.0.attn_q.bias") || has("blk.0.attn_qkv.bias"));
    has_attn_out_bias = p.attention.output_bias.value_or(has("blk.0.attn_output.bias"));
    has_attn_gate = p.attention.output_gate.value_or(has("blk.0.attn_gate.weight"));
    has_rope_freqs = has("rope_freqs.weight");
    if (!has("blk.0." + attn_post_norm_key) && has("blk.0.post_attention_norm"))
        attn_post_norm_key = "post_attention_norm";
    if (!has("blk.0." + ffn_post_norm_key) && has("blk.0.post_ffw_norm"))
        ffn_post_norm_key = "post_ffw_norm";
    has_attn_post_norm = p.normalization.attention_post.value_or(
        has("blk.0." + attn_post_norm_key) &&
        (post_norm_only || (has("blk.0." + attn_norm_key) && has("blk.0." + ffn_norm_key))));
    has_ffn_post_norm = p.normalization.ffn_post.value_or(has("blk.0." + ffn_post_norm_key) &&
                                                          (post_norm_only || has("blk.0." + ffn_norm_key)));
    if (!post_norm_only && !has("blk.0." + ffn_norm_key) && has("blk.0.post_attention_norm.weight"))
        ffn_norm_key = "post_attention_norm.weight";
    for (int layer = 0; layer < n_layer; ++layer) {
        OPENVINO_ASSERT(layer_n_head(layer) > 0 && layer_n_head_kv(layer) >= 0 && layer_head_size(layer) > 0,
                        "[GGUF] invalid per-layer dimensions at layer ",
                        layer);
        validate_rope(layer_rope_config(layer), layer_head_size(layer));
        OPENVINO_ASSERT(std::isfinite(layer_kq_scale(layer)), "[GGUF] nonfinite per-layer attention scale");
        has_swa = has_swa || layer_is_swa(layer);
        const auto local = layer_rope_config(layer);
        OPENVINO_ASSERT(local.is_imrope == rope_config.is_imrope,
                        "[GGUF] per-layer RoPE must use the shared position layout");
        use_per_op_rope = use_per_op_rope || local.per_op || local.n_ctx_orig != rope_config.n_ctx_orig ||
                          local.n_dims != rope_config.n_dims || local.freq_base != rope_config.freq_base ||
                          local.freq_scale != rope_config.freq_scale || local.ext_factor != rope_config.ext_factor ||
                          local.attn_factor != rope_config.attn_factor || local.beta_fast != rope_config.beta_fast ||
                          local.beta_slow != rope_config.beta_slow || local.sections != rope_config.sections;
    }
    use_per_op_rope = use_per_op_rope || rope_op_case == ROPE_OP_CASE_IMROPE || rope_config.per_op;
}

bool DecoderConfig::layer_is_moe(int il) const {
    if (!layers.empty() && layers.at(il).experts)
        return *layers.at(il).experts;
    return is_moe && il >= n_dense_lead && (il + 1) % moe_layer_step == 0;
}

bool DecoderConfig::layer_is_swa(int il) const {
    if (!layers.empty() && layers.at(il).sliding_window)
        return *layers.at(il).sliding_window;
    if (sliding_on_attention_layers)
        return has_swa && layer_n_head_kv(il) > 0;
    if (!swa_layer_flags.empty())
        return swa_layer_flags.at(il) != 0;
    return periodic_swa && swa_layer_pattern > 0 && il % swa_layer_pattern < swa_layer_pattern - 1;
}

int DecoderConfig::layer_head_size(int il) const {
    if (!layers.empty() && layers.at(il).head_size)
        return *layers.at(il).head_size;
    return layer_is_swa(il) && head_size_swa > 0 ? head_size_swa : head_size;
}

int DecoderConfig::layer_n_head(int il) const {
    return !layers.empty() && layers.at(il).query_heads ? *layers.at(il).query_heads : n_head;
}

int DecoderConfig::layer_n_head_kv(int il) const {
    if (!layers.empty() && layers.at(il).kv_heads)
        return *layers.at(il).kv_heads;
    return !n_head_kv_per_layer.empty() ? static_cast<int>(n_head_kv_per_layer.at(il)) : n_head_kv;
}

float DecoderConfig::layer_kq_scale(int il) const {
    if (!layers.empty() && layers.at(il).attention_scale)
        return *layers.at(il).attention_scale;
    return attention_scale != 0.0f ? attention_scale : 1.0f / std::sqrt(static_cast<float>(layer_head_size(il)));
}

RopeConfig DecoderConfig::layer_rope_config(int il) const {
    if (!layers.empty() && layers.at(il).rope)
        return *layers.at(il).rope;
    return layer_is_swa(il) ? rope_config_swa : rope_config;
}

bool DecoderConfig::is_recurrent_layer(int il) const {
    if (!layers.empty() && layers.at(il).recurrent)
        return *layers.at(il).recurrent;
    if (!is_qwen35)
        return false;
    if (!recurrent_layer_flags.empty())
        return recurrent_layer_flags.at(il) != 0;
    return full_attn_interval > 0 && (il + 1) % full_attn_interval != 0;
}

}  // namespace ov::frontend::gguf
