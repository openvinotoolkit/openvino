// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/arch/dspark_builder.hpp"

#include <cmath>
#include <limits>

#include "builder/api/metadata_store.hpp"
#include "builder/blocks/attention.hpp"
#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/decoder_config.hpp"
#include "builder/metadata_checks.hpp"

namespace ov::frontend::gguf {
namespace {
class DSparkBuilder : public ModelBuilder {
public:
    explicit DSparkBuilder(const BuildContext& context)
        : m_metadata(context.metadata),
          m_config(resolve_decoder_parameters(context.metadata, RopeMode::Neox), context.weights->weights),
          m_emitter(context.weights->weights, context.weights->qtypes, context.arch, context.weights->translators) {}

    std::shared_ptr<GgufGraph> build() override {
        using ov::element::f32;
        using ov::element::i32;
        const auto required_int = [&](const std::string& name) {
            return require_positive_int(m_metadata, "dspark.dspark." + name, "DSpark");
        };
        const auto block = required_int("block_size");
        OPENVINO_ASSERT(block <= 4096, "[GGUF] DSpark block_size exceeds the supported maximum of 4096");
        const auto markov_rank = m_metadata.get_int("dspark.dspark.markov_rank").value_or(0);
        const bool confidence_head = m_metadata.get_bool("dspark.dspark.confidence_head").value_or(false);
        const bool confidence_with_markov =
            m_metadata.get_bool("dspark.dspark.confidence_head_with_markov").value_or(false);
        OPENVINO_ASSERT(!confidence_head || !confidence_with_markov || markov_rank > 0,
                        "[GGUF] DSpark confidence_head_with_markov requires a positive markov_rank");
        const auto captures = m_metadata.get_int_array("dspark.dspark.target_layers");
        OPENVINO_ASSERT(!captures.empty(), "[GGUF] missing DSpark target_layers");
        OPENVINO_ASSERT(!m_metadata.get_bool("dspark.dspark.hidden_correction").value_or(false),
                        "[GGUF] DSpark hidden correction is not supported");
        auto& e = m_emitter;
        m_config.configure_graph(e, m_metadata);
        const auto kv = blocks::KvCachePlan::build(m_config);
        e.add_input("inp_tokens", i32, ps({1, 1, 1, block}));
        e.add_input("inp_pos", i32, ps({1, 1, 1, -1}));
        e.add_input("self_kq_mask", f32, ps({1, 1, -1, -1}));
        e.add_input("inp_kv_idx", i32, ps({1, 1, 1, -1}));
        e.add_input("token_len_per_seq", ov::element::i64, ps({1}));
        e.add_input("dspark_ctx_feat", f32, ps({1, 1, -1, static_cast<int64_t>(captures.size()) * m_config.n_embd}));
        e.add_weight("dspark.fc.weight");
        auto context = e.add_op("GGML_OP_MUL_MAT", "dspark.fc", {"dspark.fc.weight", "dspark_ctx_feat"});
        context = blocks::rms_norm(e, context, "dspark.hidden_norm.weight", "dspark.hidden_norm", m_config.rms_eps);
        e.add_weight("token_embd.weight");
        auto current = e.add_op("GGML_OP_GET_ROWS", "dspark.embedding", {"token_embd.weight", "inp_tokens"});
        if (m_metadata.get_bool("dspark.dspark.log_snr_conditioning").value_or(false)) {
            const auto low = m_metadata.get_float("dspark.dspark.min_log_snr");
            const auto high = m_metadata.get_float("dspark.dspark.max_log_snr");
            OPENVINO_ASSERT(low && high && std::isfinite(*low) && std::isfinite(*high) && *high > *low,
                            "[GGUF] invalid DSpark log-SNR bounds");
            constexpr size_t feature_width = 128;
            const auto rows = static_cast<size_t>(block);
            OPENVINO_ASSERT(rows <= std::numeric_limits<size_t>::max() / feature_width / sizeof(float),
                            "[GGUF] DSpark log-SNR feature byte count overflows size_t");
            ov::Tensor features(f32, {1, 1, rows, feature_width});
            for (size_t pos = 0; pos < rows; ++pos) {
                // Endpoint normalization maps max/min log-SNR to 1000/0 in Prism's inference schedule.
                const float normalized_time = pos == 0 ? 1000.f : 0.f;
                for (size_t i = 0; i < feature_width / 2; ++i) {
                    const float angle = normalized_time * std::exp(-std::log(10000.f) * i / 64.f);
                    features.data<float>()[pos * feature_width + i] = std::sin(angle);
                    features.data<float>()[pos * feature_width + feature_width / 2 + i] = std::cos(angle);
                }
            }
            e.weights()["dspark.log_snr_features"] = features;
            e.add_named_weight("dspark.log_snr_features");
            auto snr = std::string("dspark.log_snr_features");
            for (int layer = 1; layer <= 2; ++layer) {
                const auto name = "dspark.log_snr_fc" + std::to_string(layer);
                e.add_weight(name + ".weight");
                snr = e.add_op("GGML_OP_MUL_MAT", name, {name + ".weight", snr});
                snr = blocks::add_bias(e, snr, name + ".bias", name + ".biased");
                if (layer == 1)
                    snr = e.add_op("GGML_UNARY_OP_SILU", name + ".silu", {snr});
            }
            current = e.add_op("GGML_OP_ADD", "dspark.conditioned", {current, snr});
        }
        if (e.has_weight("dspark.mode_embedding.bias"))
            current = blocks::add_bias(e, current, "dspark.mode_embedding.bias", "dspark.mode");
        for (int layer = 0; layer < m_config.n_layer; ++layer) {
            const auto p = "blk." + std::to_string(layer) + ".";
            auto normalized = blocks::rms_norm(e, current, p + "attn_norm.weight", p + "norm", m_config.rms_eps);
            auto joined = e.add_op("GGML_OP_CONCAT", p + "context", {context, normalized}, 0, {{"concat_axis", 1}});
            auto attention = blocks::attention(e, m_config, kv, layer, joined);
            attention = e.add_op("GGML_OP_VIEW",
                                 p + "draft",
                                 {attention},
                                 3,
                                 {{"view_slice", std::vector<int64_t>{2, -block, block}}});
            current = e.add_op("GGML_OP_ADD", p + "attn_residual", {current, attention});
            normalized = blocks::rms_norm(e, current, p + "ffn_norm.weight", p + "ffn_norm", m_config.rms_eps);
            const auto ffn = blocks::dense_ffn(e, m_config, p, normalized);
            current = e.add_op("GGML_OP_ADD", p + "ffn_residual", {current, ffn});
        }
        const auto hidden = current;
        current = blocks::rms_norm(e, current, "output_norm.weight", "result_norm", m_config.rms_eps);
        const auto output = e.has_weight("output.weight") ? "output.weight" : "token_embd.weight";
        e.add_weight(output);
        const auto logits = e.add_op("GGML_OP_MUL_MAT", "logits", {output, current});
        auto logits_value = e.value(logits);
        logits_value.set_names({"logits"});
        auto hidden_value = e.value(hidden);
        hidden_value.set_names({"dspark.hidden"});
        auto& graph = *e.graph();
        graph.model_output_names.insert(graph.model_output_names.begin(), {logits, hidden});
        std::string latent;
        if (markov_rank > 0) {
            e.add_input("dspark_prev_tokens", i32, ps({1, 1, 1, block}));
            e.add_weight("dspark.markov_head_a.weight");
            e.add_weight("dspark.markov_head_b.weight");
            latent = e.add_op("GGML_OP_GET_ROWS",
                              "dspark.markov_latent",
                              {"dspark.markov_head_a.weight", "dspark_prev_tokens"});
            const auto adjustment =
                e.add_op("GGML_OP_MUL_MAT", "dspark.markov_logits", {"dspark.markov_head_b.weight", latent});
            const auto corrected = e.add_op("GGML_OP_ADD", "dspark.corrected_logits", {logits, adjustment});
            auto corrected_value = e.value(corrected);
            corrected_value.set_names({"dspark.corrected_logits"});
            graph.model_output_names.push_back(corrected);
        }
        if (confidence_head) {
            auto features = hidden;
            if (confidence_with_markov)
                features =
                    e.add_op("GGML_OP_CONCAT", "dspark.confidence_features", {hidden, latent}, 0, {{"concat_axis", 0}});
            e.add_weight("dspark.confidence_head.weight");
            auto confidence =
                e.add_op("GGML_OP_MUL_MAT", "dspark.confidence", {"dspark.confidence_head.weight", features});
            confidence = blocks::add_bias(e, confidence, "dspark.confidence_head.bias", "dspark.confidence_score");
            auto confidence_value = e.value(confidence);
            confidence_value.set_names({"dspark.confidence_score"});
            graph.model_output_names.push_back(confidence);
        }
        e.validate_hadamard();
        return e.graph();
    }

private:
    GgufMetadata m_metadata;
    DecoderConfig m_config;
    GraphEmitter m_emitter;
};
}  // namespace

std::shared_ptr<ModelBuilder> make_dspark_builder(const BuildContext& context) {
    return std::make_shared<DSparkBuilder>(context);
}
}  // namespace ov::frontend::gguf
