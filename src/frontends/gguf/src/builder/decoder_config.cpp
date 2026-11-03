// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/decoder_config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/arch_registry.hpp"
#include "builder/graph_emitter.hpp"
#include "builder/metadata_checks.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {
namespace {
bool same_rope(const RopeConfig& a, const RopeConfig& b) {
    return a.per_op == b.per_op && a.n_ctx_orig == b.n_ctx_orig && a.n_dims == b.n_dims && a.freq_base == b.freq_base &&
           a.freq_scale == b.freq_scale && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow && a.sections == b.sections;
}

void validate_rope(const RopeConfig& rope, int head_size) {
    OPENVINO_ASSERT(rope.n_dims >= 0 && rope.n_dims % 2 == 0 && rope.n_dims <= head_size,
                    "[GGUF] RoPE dimensions must be nonnegative, even and fit the attention head");
    if (rope.n_dims > 0) {
        require_positive_finite(rope.freq_base, "RoPE frequency base");
        require_positive_finite(rope.freq_scale, "RoPE frequency scale");
    }
}
}  // namespace

DecoderConfig::DecoderConfig(const DecoderParameters& p, const std::unordered_map<std::string, ov::Tensor>& weights) {
    const auto has = [&weights](const std::string& name) {
        return weights.count(name) > 0;
    };
    arch = p.architecture;
    model_contract = p.model_contract;
    n_layer = p.dimensions.layers;
    n_embd = p.dimensions.embedding;
    const int query_heads = p.dimensions.query_heads;
    const int kv_heads = p.dimensions.kv_heads;
    head_size = p.dimensions.head_size;
    const auto& kv_heads_per_layer = p.dimensions.kv_heads_per_layer;
    n_embd_per_layer = p.dimensions.per_layer_embedding;
    shared_kv_layers = p.dimensions.shared_kv_layers;
    const int n_layer_nextn = p.hybrid.nextn_layers;
    OPENVINO_ASSERT(n_layer > 0 && n_embd > 0 && query_heads > 0 && kv_heads >= 0 && head_size > 0,
                    "[GGUF] invalid decoder dimensions");
    OPENVINO_ASSERT(n_layer_nextn >= 0 && n_layer_nextn < n_layer,
                    "[GGUF] invalid number of NextN blocks: ",
                    n_layer_nextn);
    const int stored_layers = n_layer;
    n_layer -= n_layer_nextn;
    const auto valid_layer_count = [&](size_t count) {
        return count == 0 || count == static_cast<size_t>(n_layer) || count == static_cast<size_t>(stored_layers);
    };
    OPENVINO_ASSERT(valid_layer_count(p.layers.size()) && valid_layer_count(kv_heads_per_layer.size()) &&
                        valid_layer_count(p.attention.sliding_layers.size()) &&
                        valid_layer_count(p.hybrid.recurrent_layers.size()),
                    "[GGUF] per-layer settings must describe every decoder layer");
    rms_eps = p.normalization.epsilon;
    require_positive_finite(rms_eps, "normalization epsilon");
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
    if (post_norm_eps != 0.0f)
        require_positive_finite(post_norm_eps, "post-norm epsilon");
    attn_norm_key = p.normalization.attention_key;
    ffn_norm_key = p.normalization.ffn_key;
    attn_post_norm_key = p.normalization.attention_post_key;
    ffn_post_norm_key = p.normalization.ffn_post_key;
    const bool gated_delta_net = p.hybrid.kind == RecurrentKind::GatedDeltaNet;
    ssm_conv_kernel = p.hybrid.conv_kernel;
    ssm_state_size = p.hybrid.state_size;
    ssm_group_count = p.hybrid.group_count;
    ssm_dt_rank = p.hybrid.time_step_rank;
    ssm_inner_size = p.hybrid.inner_size;
    n_expert = p.experts.count;
    n_expert_used = p.experts.used;
    const int dense_lead = p.experts.dense_lead;
    const int moe_layer_step = std::max(1, p.experts.layer_step);
    expert_groups = std::max(1, p.experts.groups);
    expert_groups_used = std::max(1, p.experts.groups_used);
    moe_sigmoid_gating = p.experts.routing == ExpertRouting::Sigmoid;
    expert_weights_scale = p.experts.weights_scale;
    expert_weights_norm = p.experts.normalize_weights;
    embedding_scale = p.scales.embedding;
    residual_scale = p.scales.residual;
    logit_scale = p.scales.logits;
    embedding_scale_tokens_only = p.scales.token_embeddings_only;
    const float attention_scale = p.scales.attention;
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
    const int head_size_swa = p.attention.sliding_head_size;
    swa_window_size = p.attention.window_size;
    swa_layer_pattern = p.attention.sliding_period;
    const auto& swa_layer_flags = p.attention.sliding_layers;
    has_swa = p.attention.sliding_window || swa_window_size > 0 || !swa_layer_flags.empty();
    rope_on_swa_only = p.attention.rope_on_sliding_only;
    rope_skip_period = p.attention.rope_skip_period;
    OPENVINO_ASSERT(rope_skip_period >= 0, "[GGUF] RoPE skip period must be nonnegative");
    OPENVINO_ASSERT(swa_window_size >= 0 && swa_layer_pattern >= 0, "[GGUF] invalid sliding-window settings");
    rope_config = p.attention.rope;
    auto rope_config_swa = p.attention.sliding_rope;
    if (rope_config_swa.n_dims == 0)
        rope_config_swa = rope_config;
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
        rope_config_swa.is_imrope = true;
        rope_config_swa.sections = rope_config.sections;
    }
    OPENVINO_ASSERT(rope_config.is_imrope == (rope_op_case == ROPE_OP_CASE_IMROPE),
                    "[GGUF] RoPE configuration must match the selected position layout");
    has_sinks = p.attention.sinks.value_or(has("blk.0.attn_sinks.weight"));
    has_swa = has_swa || has_sinks;
    const bool periodic_swa = has_swa;
    const int probe = (std::max(0, dense_lead) / moe_layer_step + 1) * moe_layer_step - 1;
    const auto expert_prefix = "blk." + std::to_string(probe) + ".";
    const bool is_moe = p.experts.enabled.value_or(has(expert_prefix + "ffn_gate_exps.weight") ||
                                                   has(expert_prefix + "ffn_gate_up_exps.weight"));
    const auto layer_overrides = [&p](int il) -> const DecoderLayerOverrides* {
        return p.layers.empty() ? nullptr : &p.layers.at(il);
    };
    const auto periodic_or_listed_swa = [&](int il) {
        if (!swa_layer_flags.empty())
            return swa_layer_flags.at(il) != 0;
        return periodic_swa && swa_layer_pattern > 0 && il % swa_layer_pattern < swa_layer_pattern - 1;
    };
    // Layers flagged by explicit overrides or the period/flag rules decide whether the model uses
    // SWA at all; attention-only SWA layouts then follow it.
    for (int il = 0; il < n_layer; ++il) {
        const auto* o = layer_overrides(il);
        has_swa = has_swa ||
                  (o && o->sliding_window ? *o->sliding_window
                                          : !p.attention.sliding_on_attention_layers && periodic_or_listed_swa(il));
    }
    m_layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const auto* o = layer_overrides(il);
        auto& layer = m_layers[il];
        layer.n_head = o && o->query_heads ? *o->query_heads : query_heads;
        layer.n_head_kv = o && o->kv_heads              ? *o->kv_heads
                          : !kv_heads_per_layer.empty() ? static_cast<int>(kv_heads_per_layer.at(il))
                                                        : kv_heads;
        layer.swa = o && o->sliding_window                    ? *o->sliding_window
                    : p.attention.sliding_on_attention_layers ? has_swa && layer.n_head_kv > 0
                                                              : periodic_or_listed_swa(il);
        layer.head_size = o && o->head_size                ? *o->head_size
                          : layer.swa && head_size_swa > 0 ? head_size_swa
                                                           : head_size;
        layer.kq_scale = o && o->attention_scale   ? *o->attention_scale
                         : attention_scale != 0.0f ? attention_scale
                                                   : 1.0f / std::sqrt(static_cast<float>(layer.head_size));
        layer.rope = o && o->rope ? *o->rope : layer.swa ? rope_config_swa : rope_config;
        layer.moe = o && o->experts ? *o->experts : is_moe && il >= dense_lead && (il + 1) % moe_layer_step == 0;
        layer.recurrent = o && o->recurrent                    ? *o->recurrent
                          : !gated_delta_net                   ? false
                          : !p.hybrid.recurrent_layers.empty() ? p.hybrid.recurrent_layers.at(il) != 0
                                                               : p.hybrid.full_attention_interval > 0 &&
                                                                     (il + 1) % p.hybrid.full_attention_interval != 0;
    }
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
    has_moe_gate_bias = has(expert_prefix + "ffn_gate_inp.bias");
    has_moe_expert_bias = has(expert_prefix + "ffn_gate_exps.bias");
    moe_softmax_weight = p.experts.softmax_after_topk.value_or(has_moe_gate_bias);
    moe_swiglu_oai = p.experts.swiglu_oai.value_or(has_moe_gate_bias);
    interleaved_q_gate = p.attention.interleaved_q_gate;
    qkv_bias_override = p.attention.qkv_bias;
    output_bias_override = p.attention.output_bias;
    fused_qkv_override = p.attention.fused_qkv;
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
        const auto& settings = m_layers[layer];
        OPENVINO_ASSERT(settings.n_head > 0 && settings.n_head_kv >= 0 && settings.head_size > 0,
                        "[GGUF] invalid per-layer dimensions at layer ",
                        layer);
        validate_rope(settings.rope, settings.head_size);
        OPENVINO_ASSERT(std::isfinite(settings.kq_scale), "[GGUF] nonfinite per-layer attention scale");
        OPENVINO_ASSERT(settings.rope.is_imrope == rope_config.is_imrope,
                        "[GGUF] per-layer RoPE must use the shared position layout");
        use_per_op_rope = use_per_op_rope || !same_rope(settings.rope, rope_config);
    }
    use_per_op_rope = use_per_op_rope || rope_op_case == ROPE_OP_CASE_IMROPE || rope_config.per_op;
}

void DecoderConfig::configure_graph(GraphEmitter& emitter, const GgufMetadata& metadata) const {
    emitter.configure_hadamard(decoder_weight_transform_metadata(metadata, ssm_dt_rank, ssm_group_count));
    auto& graph = *emitter.graph();
    graph.has_rope = true;
    graph.rope_config = rope_config;
    graph.use_per_op_rope = use_per_op_rope;
    graph.swa_window_size = swa_window_size;
    graph.model_contract = model_contract;
}
}  // namespace ov::frontend::gguf
