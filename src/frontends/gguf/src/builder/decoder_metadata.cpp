// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/decoder_metadata.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/api/metadata_store.hpp"
#include "builder/arch/decoder_builder.hpp"
#include "builder/metadata_checks.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {
namespace {
int integer(const GgufMetadata& m, const std::string& key, int fallback, bool required = false) {
    if (!m.has(key)) {
        OPENVINO_ASSERT(!required, "[GGUF] required metadata key is missing: '", key, "'");
        return fallback;
    }
    const auto value = m.get_int(key);
    OPENVINO_ASSERT(value && *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max(),
                    "[GGUF] invalid integer metadata key '",
                    key,
                    "'");
    return static_cast<int>(*value);
}
float real(const GgufMetadata& m, const std::string& key, float fallback, bool required = false) {
    if (!m.has(key)) {
        OPENVINO_ASSERT(!required, "[GGUF] required metadata key is missing: '", key, "'");
        return fallback;
    }
    const auto value = m.get_float(key);
    OPENVINO_ASSERT(value && std::isfinite(*value) && std::abs(*value) <= std::numeric_limits<float>::max(),
                    "[GGUF] invalid floating-point metadata key '",
                    key,
                    "'");
    return static_cast<float>(*value);
}
bool flag(const GgufMetadata& m, const std::string& key, bool fallback) {
    if (!m.has(key))
        return fallback;
    const auto value = m.get_bool(key);
    OPENVINO_ASSERT(value, "[GGUF] invalid boolean metadata key '", key, "'");
    return *value;
}
std::vector<int32_t> integers(const GgufMetadata& m, const std::string& key) {
    if (!m.has(key))
        return {};
    const auto values = m.get_int_array(key);
    OPENVINO_ASSERT(!values.empty(), "[GGUF] invalid integer array metadata key '", key, "'");
    std::vector<int32_t> result;
    result.reserve(values.size());
    for (const auto value : values) {
        OPENVINO_ASSERT(value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
                        "[GGUF] integer array metadata key '",
                        key,
                        "' exceeds int32 range");
        result.push_back(static_cast<int32_t>(value));
    }
    return result;
}
}  // namespace

DecoderParameters read_decoder_parameters(const GgufMetadata& m, DecoderParameters p) {
    const auto architecture = m.get_str("general.architecture");
    OPENVINO_ASSERT(architecture, "[GGUF] missing or non-string general.architecture");
    p.architecture = *architecture;
    const auto key = [&](const std::string& suffix) {
        return p.architecture + "." + suffix;
    };
    auto& d = p.dimensions;
    d.layers = integer(m, key("block_count"), d.layers, d.layers == 0);
    d.embedding = integer(m, key("embedding_length"), d.embedding, d.embedding == 0);
    const auto query_key = key("attention.head_count");
    std::vector<int32_t> query_heads;
    if (m.has(query_key) && !m.get_int(query_key)) {
        query_heads = integers(m, query_key);
        OPENVINO_ASSERT(d.layers > 0 && query_heads.size() == static_cast<size_t>(d.layers),
                        "[GGUF] query head counts must describe every decoder layer");
        d.query_heads = *std::max_element(query_heads.begin(), query_heads.end());
        OPENVINO_ASSERT(p.layers.empty() || p.layers.size() == query_heads.size(),
                        "[GGUF] per-layer settings must describe every decoder layer");
        p.layers.resize(query_heads.size());
        for (size_t layer = 0; layer < query_heads.size(); ++layer) {
            OPENVINO_ASSERT(query_heads[layer] >= 0, "[GGUF] negative query head count");
            if (query_heads[layer] > 0)
                p.layers[layer].query_heads = query_heads[layer];
        }
    } else {
        d.query_heads = integer(m, query_key, d.query_heads, d.query_heads == 0);
    }
    OPENVINO_ASSERT(d.layers > 0 && d.embedding > 0 && d.query_heads > 0,
                    "[GGUF] decoder dimensions and attention.head_count must be positive");
    d.head_size = integer(m, key("attention.key_length"), d.head_size > 0 ? d.head_size : d.embedding / d.query_heads);
    const auto kv_key = key("attention.head_count_kv");
    if (m.has(kv_key) && !m.get_int(kv_key)) {
        const auto heads = integers(m, kv_key);
        d.kv_heads_per_layer.clear();
        for (const auto count : heads) {
            OPENVINO_ASSERT(count >= 0, "[GGUF] negative per-layer KV head count");
            d.kv_heads_per_layer.push_back(static_cast<uint32_t>(count));
        }
        d.kv_heads = static_cast<int>(*std::max_element(heads.begin(), heads.end()));
    } else {
        d.kv_heads = integer(m, kv_key, d.kv_heads > 0 ? d.kv_heads : d.query_heads);
    }
    const auto norm_key = key(p.normalization.kind == NormalizationKind::Layer ? "attention.layer_norm_epsilon"
                                                                               : "attention.layer_norm_rms_epsilon");
    p.normalization.epsilon = real(m, norm_key, p.normalization.epsilon, p.normalization.epsilon == 0.0f);
    const auto pooling_key = key("pooling_type");
    const auto pooling = m.get_int(pooling_key);
    // llama.cpp stores its unset pooling enum as an unsigned -1.
    p.pooling_type =
        pooling && *pooling == std::numeric_limits<uint32_t>::max() ? -1 : integer(m, pooling_key, p.pooling_type);
    p.causal_attention = flag(m, key("attention.causal"), p.causal_attention);
    auto& a = p.attention;
    a.rope.freq_base = real(m, key("rope.freq_base"), a.rope.freq_base > 0 ? a.rope.freq_base : 10000.0f);
    const float factor = real(m, key("rope.scaling.factor"), 0.0f);
    a.rope.freq_scale = a.rope.freq_scale > 0 ? a.rope.freq_scale : 1.0f;
    if (factor != 0.0f) {
        OPENVINO_ASSERT(factor > 0.0f, "[GGUF] RoPE scaling factor must be finite and positive");
        const double inverse = 1.0 / static_cast<double>(factor);
        OPENVINO_ASSERT(inverse <= std::numeric_limits<float>::max(),
                        "[GGUF] RoPE factor reciprocal exceeds float range");
        a.rope.freq_scale = static_cast<float>(inverse);
    }
    if (m.has(key("rope.scaling.type"))) {
        const auto type = m.get_str(key("rope.scaling.type"));
        OPENVINO_ASSERT(type, "[GGUF] invalid rope.scaling.type");
        a.rope.ext_factor = *type == "yarn" ? 1.0f : 0.0f;
    }
    a.rope.beta_fast = real(m, key("rope.scaling.yarn_beta_fast"), a.rope.beta_fast > 0 ? a.rope.beta_fast : 32.0f);
    a.rope.beta_slow = real(m, key("rope.scaling.yarn_beta_slow"), a.rope.beta_slow > 0 ? a.rope.beta_slow : 1.0f);
    a.yarn_log_multiplier = real(m, key("rope.scaling.yarn_log_multiplier"), a.yarn_log_multiplier);
    a.rope.attn_factor = a.rope.attn_factor == 0.0f ? 1.0f : a.rope.attn_factor;
    const int context = integer(m, key("context_length"), 2048);
    a.rope.n_ctx_orig =
        integer(m, key("rope.scaling.original_context_length"), a.rope.n_ctx_orig > 0 ? a.rope.n_ctx_orig : context);
    const int rope_dims = integer(m, key("rope.dimension_count"), a.rope.n_dims);
    a.rope.n_dims = rope_dims > 0 ? rope_dims : d.head_size;
    a.sliding_head_size =
        integer(m, key("attention.key_length_swa"), a.sliding_head_size > 0 ? a.sliding_head_size : d.head_size);
    a.sliding_rope.n_dims =
        integer(m, key("rope.dimension_count_swa"), a.sliding_rope.n_dims > 0 ? a.sliding_rope.n_dims : a.rope.n_dims);
    a.sliding_rope.freq_base =
        real(m, key("rope.freq_base_swa"), a.sliding_rope.freq_base > 0 ? a.sliding_rope.freq_base : a.rope.freq_base);
    if (a.sliding_rope.freq_scale == 0.0f)
        a.sliding_rope.freq_scale = a.rope.freq_scale;
    a.sliding_rope.n_ctx_orig = a.rope.n_ctx_orig;
    a.sliding_rope.ext_factor = a.rope.ext_factor;
    a.sliding_rope.attn_factor = a.rope.attn_factor;
    a.sliding_rope.beta_fast = a.rope.beta_fast;
    a.sliding_rope.beta_slow = a.rope.beta_slow;
    if (m.has(key("attention.sliding_window"))) {
        const auto value = m.get_int(key("attention.sliding_window"));
        OPENVINO_ASSERT(value && *value >= 0 && *value <= std::numeric_limits<uint32_t>::max(),
                        "[GGUF] invalid attention.sliding_window");
        if (*value > 0 && *value != std::numeric_limits<uint32_t>::max()) {
            OPENVINO_ASSERT(*value <= std::numeric_limits<int>::max(), "[GGUF] sliding window exceeds int range");
            a.window_size = static_cast<int>(*value);
        }
    }
    const auto pattern = key("attention.sliding_window_pattern");
    if (m.has(pattern)) {
        const auto& stored_pattern = detail::MetadataAccess::get(m).map.at(pattern);
        const auto* tensor = std::get_if<ov::Tensor>(&stored_pattern);
        const bool boolean_pattern = tensor && tensor->get_element_type() == ov::element::boolean;
        if (!boolean_pattern && m.get_int(pattern)) {
            a.sliding_period = integer(m, pattern, a.sliding_period);
            a.sliding_layers.clear();
        } else {
            a.sliding_layers = integers(m, pattern);
            a.sliding_period = 0;
        }
        a.sliding_window = true;
    }
    a.sliding_window = a.sliding_window || a.window_size > 0 || !a.sliding_layers.empty();
    d.per_layer_embedding = integer(m, key("embedding_length_per_layer_input"), d.per_layer_embedding);
    d.shared_kv_layers = integer(m, key("attention.shared_kv_layers"), d.shared_kv_layers);
    auto& e = p.experts;
    e.count = integer(m, key("expert_count"), e.count);
    e.used = integer(m, key("expert_used_count"), e.used);
    e.dense_lead = integer(m, key("leading_dense_block_count"), e.dense_lead);
    e.layer_step = integer(m, key("interleave_moe_layer_step"), e.layer_step);
    e.groups = integer(m, key("expert_group_count"), e.groups);
    e.groups_used = integer(m, key("expert_group_used_count"), e.groups_used);
    // Zero is llama.cpp's "unspecified" gating function; keep the family default.
    if (const int gating = integer(m, key("expert_gating_func"), 0))
        e.routing = static_cast<ExpertRouting>(gating);
    e.weights_scale = real(m, key("expert_weights_scale"), e.weights_scale);
    e.normalize_weights = flag(m, key("expert_weights_norm"), e.normalize_weights);
    auto& scales = p.scales;
    scales.embedding = real(m, key("embedding_scale"), scales.embedding);
    scales.residual = real(m, key("residual_scale"), scales.residual);
    // llama.cpp treats zero metadata scales as unspecified, with no multiplication.
    if (scales.embedding == 0.0f && m.has(key("embedding_scale")))
        scales.embedding = 1.0f;
    if (scales.residual == 0.0f && m.has(key("residual_scale")))
        scales.residual = 1.0f;
    scales.logits = real(m, key("logit_scale"), scales.logits);
    scales.attention = real(m, key("attention.scale"), scales.attention);
    scales.attention_temperature = real(m, key("attention.temperature_scale"), scales.attention_temperature);
    scales.attention_soft_cap = real(m, key("attn_logit_softcapping"), scales.attention_soft_cap);
    scales.logits_soft_cap = real(m, key("final_logit_softcapping"), scales.logits_soft_cap);
    auto& hybrid = p.hybrid;
    hybrid.conv_kernel = integer(m, key("ssm.conv_kernel"), hybrid.conv_kernel);
    hybrid.state_size = integer(m, key("ssm.state_size"), hybrid.state_size);
    hybrid.group_count = integer(m, key("ssm.group_count"), hybrid.group_count);
    hybrid.time_step_rank = integer(m, key("ssm.time_step_rank"), hybrid.time_step_rank);
    hybrid.inner_size = integer(m, key("ssm.inner_size"), hybrid.inner_size);
    hybrid.full_attention_interval = integer(m, key("full_attention_interval"), hybrid.full_attention_interval);
    hybrid.nextn_layers = integer(m, key("nextn_predict_layers"), hybrid.nextn_layers);
    const auto recurrent_key = key("attention.recurrent_layers");
    if (m.has(recurrent_key)) {
        if (const auto value = m.get_int(recurrent_key)) {
            OPENVINO_ASSERT(*value == 0 || *value == 1, "[GGUF] invalid recurrent layer flag");
            hybrid.recurrent_layers.assign(d.layers, static_cast<int32_t>(*value));
        } else {
            hybrid.recurrent_layers = integers(m, recurrent_key);
        }
    }
    for (size_t layer = 0; layer < query_heads.size(); ++layer)
        OPENVINO_ASSERT(query_heads[layer] > 0 ||
                            (hybrid.recurrent_layers.size() == query_heads.size() && hybrid.recurrent_layers[layer]) ||
                            (d.kv_heads_per_layer.size() == query_heads.size() && d.kv_heads_per_layer[layer] == 0),
                        "[GGUF] zero query head count requires a layer without attention");
    if (m.has(key("rope.dimension_sections"))) {
        const auto sections = integers(m, key("rope.dimension_sections"));
        OPENVINO_ASSERT(sections.size() <= a.rope.sections.size(), "[GGUF] too many M-RoPE sections");
        std::copy(sections.begin(), sections.end(), a.rope.sections.begin());
    }
    return p;
}

void apply_decoder_options(DecoderParameters& p, const DecoderOptions& o) {
    if (o.normalization)
        p.normalization.kind = *o.normalization;
    if (o.norm_epsilon)
        p.normalization.epsilon = *o.norm_epsilon;
    if (o.embedding_scale)
        p.scales.embedding = *o.embedding_scale;
    if (o.residual_scale)
        p.scales.residual = *o.residual_scale;
    if (o.logit_scale)
        p.scales.logits = *o.logit_scale;
    if (o.attention_scale)
        p.scales.attention = *o.attention_scale;
    if (o.qk_norm_after_rope)
        p.normalization.qk_after_rope = *o.qk_norm_after_rope;
    if (o.post_norm_only)
        p.normalization.post_only = *o.post_norm_only;
    if (o.normalize_expert_weights)
        p.experts.normalize_weights = *o.normalize_expert_weights;
    if (o.value_norm)
        p.normalization.value_norm = *o.value_norm;
    if (o.embedding_norm)
        p.normalization.embedding_norm = *o.embedding_norm;
    if (o.rope_on_swa_only)
        p.attention.rope_on_sliding_only = *o.rope_on_swa_only;
    if (o.rope_skip_period)
        p.attention.rope_skip_period = *o.rope_skip_period;
    if (o.geglu)
        p.activation = *o.geglu ? FfnActivation::GeGlu : FfnActivation::SwiGlu;
    if (o.post_norm_epsilon) {
        require_positive_finite(*o.post_norm_epsilon, "post-norm epsilon");
        p.normalization.post_epsilon = *o.post_norm_epsilon;
    }
    if (o.swa_rope_frequency_base) {
        require_positive_finite(*o.swa_rope_frequency_base, "SWA RoPE frequency base");
        p.attention.sliding_rope.freq_base = *o.swa_rope_frequency_base;
    }
    if (o.swa_rope_dimensions) {
        const int head_size =
            p.attention.sliding_head_size > 0 ? p.attention.sliding_head_size : p.dimensions.head_size;
        OPENVINO_ASSERT(
            *o.swa_rope_dimensions > 0 && *o.swa_rope_dimensions % 2 == 0 && *o.swa_rope_dimensions <= head_size,
            "[GGUF] SWA RoPE dimensions must be positive, even and fit the attention head");
        p.attention.sliding_rope.n_dims = *o.swa_rope_dimensions;
    }
    if (o.sliding_window) {
        OPENVINO_ASSERT(*o.sliding_window > 0, "[GGUF] sliding window must be positive");
        p.attention.window_size = *o.sliding_window;
        p.attention.sliding_window = true;
    }
}

std::map<std::string, GGUFMetaData> decoder_weight_transform_metadata(const GgufMetadata& metadata,
                                                                      int value_heads,
                                                                      int key_heads) {
    std::map<std::string, GGUFMetaData> result;
    for (const auto& [key, value] : detail::MetadataAccess::get(metadata).map)
        if (key.compare(0, 15, "prism.hadamard.") == 0)
            result[key] = value;
    result["ssm_time_step_rank"] = value_heads;
    result["ssm_group_count"] = key_heads;
    return result;
}

ArchitectureDefinition::BuilderFactory decoder_builder_factory(DecoderParametersFn parameters) {
    OPENVINO_ASSERT(parameters, "[GGUF] decoder architecture requires a parameter factory");
    return [parameters = std::move(parameters)](const BuildContext& ctx) {
        OPENVINO_ASSERT(ctx.weights, "[GGUF] decoder builder requires a weight table");
        return std::make_shared<DecoderBuilder>(ctx, parameters(ctx));
    };
}

ArchitectureDefinition::BuilderFactory decoder_builder_factory(RopeMode rope, DecoderOptionsFn options) {
    return decoder_builder_factory(DecoderParametersFn([rope, options = std::move(options)](const BuildContext& ctx) {
        return resolve_decoder_parameters(ctx.metadata, rope, options ? options(ctx.metadata) : DecoderOptions{});
    }));
}

}  // namespace ov::frontend::gguf
