// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/decoder_metadata.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/api/metadata_store.hpp"
#include "builder/arch/decoder_builder.hpp"
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
    e.routing = static_cast<ExpertRouting>(integer(m, key("expert_gating_func"), static_cast<int>(e.routing)));
    e.weights_scale = real(m, key("expert_weights_scale"), e.weights_scale);
    e.normalize_weights = flag(m, key("expert_weights_norm"), false) || e.normalize_weights;
    auto& scales = p.scales;
    scales.embedding = real(m, key("embedding_scale"), scales.embedding);
    scales.residual = real(m, key("residual_scale"), scales.residual);
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
                            (hybrid.recurrent_layers.size() == query_heads.size() && hybrid.recurrent_layers[layer]),
                        "[GGUF] zero query head count requires a recurrent layer");
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
    if (o.post_norm_epsilon)
        p.normalization.post_epsilon = *o.post_norm_epsilon;
    if (o.swa_rope_frequency_base)
        p.attention.sliding_rope.freq_base = *o.swa_rope_frequency_base;
    if (o.swa_rope_dimensions)
        p.attention.sliding_rope.n_dims = *o.swa_rope_dimensions;
    if (o.sliding_window)
        p.attention.window_size = *o.sliding_window;
    if (o.rope_skip_period)
        p.attention.rope_skip_period = *o.rope_skip_period;
    if (o.geglu)
        p.activation = *o.geglu ? FfnActivation::GeGlu : FfnActivation::SwiGlu;
    if (o.sliding_window) {
        OPENVINO_ASSERT(*o.sliding_window > 0, "[GGUF] sliding window must be positive");
        p.attention.sliding_window = true;
    }
    if (o.swa_rope_dimensions)
        OPENVINO_ASSERT(*o.swa_rope_dimensions > 0 && *o.swa_rope_dimensions % 2 == 0 &&
                            *o.swa_rope_dimensions <= (p.attention.sliding_head_size > 0 ? p.attention.sliding_head_size
                                                                                         : p.dimensions.head_size),
                        "[GGUF] SWA RoPE dimensions must be positive, even and fit the attention head");
    if (o.post_norm_epsilon)
        OPENVINO_ASSERT(std::isfinite(*o.post_norm_epsilon) && *o.post_norm_epsilon > 0,
                        "[GGUF] post-norm epsilon must be finite and positive");
    if (o.swa_rope_frequency_base)
        OPENVINO_ASSERT(std::isfinite(*o.swa_rope_frequency_base) && *o.swa_rope_frequency_base > 0,
                        "[GGUF] SWA RoPE frequency base must be finite and positive");
}

std::map<std::string, GGUFMetaData> decoder_config_from_parameters(const DecoderParameters& p) {
    std::map<std::string, GGUFMetaData> result;
    result["architecture"] = p.architecture;
    result["layer_num"] = p.dimensions.layers;
    result["head_num"] = p.dimensions.query_heads;
    result["head_num_kv"] = p.dimensions.kv_heads;
    result["head_size"] = p.dimensions.head_size;
    result["hidden_size"] = p.dimensions.embedding;
    result["rms_norm_eps"] = p.normalization.epsilon;
    result["n_layer_dense_lead"] = p.experts.dense_lead;
    result["moe_layer_step"] = p.experts.layer_step;
    result["expert_count"] = p.experts.count;
    result["expert_used_count"] = p.experts.used;
    result["expert_groups"] = p.experts.groups;
    result["expert_groups_used"] = p.experts.groups_used;
    result["ssm_conv_kernel"] = p.hybrid.conv_kernel;
    result["ssm_state_size"] = p.hybrid.state_size;
    result["ssm_group_count"] = p.hybrid.group_count;
    result["ssm_time_step_rank"] = p.hybrid.time_step_rank;
    result["ssm_inner_size"] = p.hybrid.inner_size;
    result["full_attention_interval"] = p.hybrid.full_attention_interval;
    result["nextn_predict_layers"] = p.hybrid.nextn_layers;
    result["embedding_scale"] = p.scales.embedding;
    result["residual_scale"] = p.scales.residual;
    result["logit_scale"] = p.scales.logits;
    result["attention_scale"] = p.scales.attention;
    result["expert_weights_scale"] = p.experts.weights_scale;
    result["rope_freq_base_swa"] = p.attention.sliding_rope.freq_base;
    result["swa_layer_pattern"] = p.attention.sliding_period;
    result["attn_logit_softcapping"] = p.scales.attention_soft_cap;
    result["final_logit_softcapping"] = p.scales.logits_soft_cap;
    result["n_embd_per_layer"] = p.dimensions.per_layer_embedding;
    result["shared_kv_layers"] = p.dimensions.shared_kv_layers;
    result["head_size_swa"] = p.attention.sliding_head_size;
    result["rope_dimension_count_swa"] = p.attention.sliding_rope.n_dims;
    result["swa_window_size"] = p.attention.window_size;
    result["rope_dimension_count"] = p.attention.rope.n_dims;
    result["rope_n_ctx_orig"] = p.attention.rope.n_ctx_orig;
    result["rope_freq_base"] = p.attention.rope.freq_base;
    result["rope_freq_scale"] = p.attention.rope.freq_scale;
    result["rope_ext_factor"] = p.attention.rope.ext_factor;
    result["rope_yarn_beta_fast"] = p.attention.rope.beta_fast;
    result["rope_yarn_beta_slow"] = p.attention.rope.beta_slow;
    result["rope_yarn_log_mul"] = p.attention.yarn_log_multiplier;
    result["attention_temperature_scale"] = p.scales.attention_temperature;
    result["pooling_type"] = p.pooling_type;
    result["causal_attention"] = p.causal_attention ? 1 : 0;
    result["expert_weights_norm"] = p.experts.normalize_weights ? 1 : 0;
    result["expert_gating_func"] = static_cast<int>(p.experts.routing);
    result["has_swa"] = p.attention.sliding_window ? 1 : 0;
    result["swa_layer_flags"] = p.attention.sliding_layers;
    result["recurrent_layer_flags"] = p.hybrid.recurrent_layers;
    result["rope_sections"] = std::vector<int32_t>(p.attention.rope.sections.begin(), p.attention.rope.sections.end());
    if (!p.dimensions.kv_heads_per_layer.empty()) {
        ov::Tensor counts(ov::element::u32, {p.dimensions.kv_heads_per_layer.size()});
        std::copy(p.dimensions.kv_heads_per_layer.begin(),
                  p.dimensions.kv_heads_per_layer.end(),
                  counts.data<uint32_t>());
        result["head_num_kv_per_layer"] = counts;
    }
    return result;
}

DecoderParameters decoder_parameters_from_config(const std::map<std::string, GGUFMetaData>& c) {
    const auto i = [&](const std::string& key) {
        return std::get<int>(c.at(key));
    };
    const auto f = [&](const std::string& key) {
        return std::get<float>(c.at(key));
    };
    auto p = decoder_family_defaults(std::get<std::string>(c.at("architecture")),
                                     i("layer_num"),
                                     i("hidden_size"),
                                     i("head_num"));
    p.dimensions.layers = i("layer_num");
    p.dimensions.query_heads = i("head_num");
    p.dimensions.kv_heads = i("head_num_kv");
    p.dimensions.head_size = i("head_size");
    p.dimensions.embedding = i("hidden_size");
    p.normalization.epsilon = f("rms_norm_eps");
    p.experts.dense_lead = i("n_layer_dense_lead");
    p.experts.layer_step = i("moe_layer_step");
    p.experts.count = i("expert_count");
    p.experts.used = i("expert_used_count");
    p.experts.groups = i("expert_groups");
    p.experts.groups_used = i("expert_groups_used");
    p.hybrid.conv_kernel = i("ssm_conv_kernel");
    p.hybrid.state_size = i("ssm_state_size");
    p.hybrid.group_count = i("ssm_group_count");
    p.hybrid.time_step_rank = i("ssm_time_step_rank");
    p.hybrid.inner_size = i("ssm_inner_size");
    p.hybrid.full_attention_interval = i("full_attention_interval");
    p.hybrid.nextn_layers = i("nextn_predict_layers");
    p.scales.embedding = f("embedding_scale");
    p.scales.residual = f("residual_scale");
    p.scales.logits = f("logit_scale");
    p.scales.attention = f("attention_scale");
    p.experts.weights_scale = f("expert_weights_scale");
    p.attention.sliding_rope.freq_base = f("rope_freq_base_swa");
    p.attention.sliding_period = i("swa_layer_pattern");
    p.scales.attention_soft_cap = f("attn_logit_softcapping");
    p.scales.logits_soft_cap = f("final_logit_softcapping");
    p.dimensions.per_layer_embedding = i("n_embd_per_layer");
    p.dimensions.shared_kv_layers = i("shared_kv_layers");
    p.attention.sliding_head_size = i("head_size_swa");
    p.attention.sliding_rope.n_dims = i("rope_dimension_count_swa");
    p.attention.window_size = i("swa_window_size");
    p.attention.rope.n_dims = i("rope_dimension_count");
    p.attention.rope.n_ctx_orig = i("rope_n_ctx_orig");
    p.attention.rope.freq_base = f("rope_freq_base");
    p.attention.rope.freq_scale = f("rope_freq_scale");
    p.attention.rope.ext_factor = f("rope_ext_factor");
    p.attention.rope.beta_fast = f("rope_yarn_beta_fast");
    p.attention.rope.beta_slow = f("rope_yarn_beta_slow");
    p.attention.yarn_log_multiplier = f("rope_yarn_log_mul");
    p.scales.attention_temperature = f("attention_temperature_scale");
    p.pooling_type = i("pooling_type");
    p.causal_attention = i("causal_attention") != 0;
    p.experts.normalize_weights = i("expert_weights_norm") != 0;
    p.experts.routing = static_cast<ExpertRouting>(i("expert_gating_func"));
    p.attention.sliding_window = i("has_swa") != 0;
    p.attention.sliding_layers = std::get<std::vector<int32_t>>(c.at("swa_layer_flags"));
    p.hybrid.recurrent_layers = std::get<std::vector<int32_t>>(c.at("recurrent_layer_flags"));
    p.attention.rope.attn_factor = 1.0f;
    const auto& sections = std::get<std::vector<int32_t>>(c.at("rope_sections"));
    OPENVINO_ASSERT(sections.size() <= p.attention.rope.sections.size(), "[GGUF] too many M-RoPE sections");
    std::copy(sections.begin(), sections.end(), p.attention.rope.sections.begin());
    p.attention.sliding_rope.n_ctx_orig = p.attention.rope.n_ctx_orig;
    p.attention.sliding_rope.ext_factor = p.attention.rope.ext_factor;
    p.attention.sliding_rope.attn_factor = p.attention.rope.attn_factor;
    p.attention.sliding_rope.beta_fast = p.attention.rope.beta_fast;
    p.attention.sliding_rope.beta_slow = p.attention.rope.beta_slow;
    if (p.attention.sliding_rope.freq_scale == 0.0f)
        p.attention.sliding_rope.freq_scale = p.attention.rope.freq_scale;
    if (c.count("head_num_kv_per_layer")) {
        const auto& t = std::get<ov::Tensor>(c.at("head_num_kv_per_layer"));
        const auto* data = t.data<const uint32_t>();
        p.dimensions.kv_heads_per_layer.assign(data, data + t.get_size());
    }
    return p;
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

std::map<std::string, GGUFMetaData> decoder_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    detail::MetadataStore store{metadata};
    const GgufMetadata view(store);
    const auto p = resolve_decoder_parameters(view, RopeMode::Normal);
    auto result = decoder_config_from_parameters(p);
    const auto transforms = decoder_weight_transform_metadata(view, p.hybrid.time_step_rank, p.hybrid.group_count);
    result.insert(transforms.begin(), transforms.end());
    result["max_position_embeddings"] = integer(view, p.architecture + ".context_length", 2048);
    result["file_type"] = integer(view, "general.file_type", 0);
    result["expert_feed_forward_length"] = integer(view, p.architecture + ".expert_feed_forward_length", 0);
    result["expert_shared_count"] = integer(view, p.architecture + ".expert_shared_count", 0);
    return result;
}

ArchitectureDefinition::BuilderFactory decoder_builder_factory(RopeMode rope, DecoderOptionsFn options) {
    return [rope, options = std::move(options)](const BuildContext& ctx) {
        OPENVINO_ASSERT(ctx.weights, "[GGUF] decoder builder requires a weight table");
        const auto overrides = options ? options(ctx.metadata) : DecoderOptions{};
        const auto parameters = resolve_decoder_parameters(ctx.metadata, rope, overrides);
        const auto builder = std::make_shared<DecoderBuilder>(ctx, parameters);
        return builder;
    };
}

ArchitectureDefinition::BuilderFactory decoder_builder_factory(DecoderParametersFn parameters) {
    OPENVINO_ASSERT(parameters, "[GGUF] decoder architecture requires a parameter factory");
    return [parameters = std::move(parameters)](const BuildContext& ctx) {
        OPENVINO_ASSERT(ctx.weights, "[GGUF] decoder builder requires a weight table");
        const auto decoder_parameters = parameters(ctx);
        const auto builder = std::make_shared<DecoderBuilder>(ctx, decoder_parameters);
        return builder;
    };
}

}  // namespace ov::frontend::gguf
